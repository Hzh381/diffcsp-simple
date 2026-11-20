"""Training and predictor-only sampling for joint crystal diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps, offset=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alpha_bar = torch.cos(
        ((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5
    ) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-4, 0.9999)


def wrapped_normal_prob(x, sigma, periods=10):
    probability = torch.zeros_like(x)
    for shift in range(-periods, periods + 1):
        value = (x + shift) / sigma
        probability = probability + torch.exp(-0.5 * value.square())
    return probability.clamp_min(1e-12)


def wrapped_normal_score(x, sigma, periods=10):
    probability = wrapped_normal_prob(x, sigma, periods)
    numerator = torch.zeros_like(x)
    for shift in range(-periods, periods + 1):
        value = x + shift
        numerator = numerator + value / sigma.square() * torch.exp(
            -0.5 * (value / sigma).square()
        )
    return numerator / probability


def estimate_score_variance(sigmas, samples=512):
    """Estimate E[||score||^2] with a small fixed Monte Carlo sample."""
    noises = torch.randn(samples, len(sigmas), 3) * sigmas[None, :, None]
    score = wrapped_normal_score(noises, sigmas[None, :, None])
    return score.square().mean(dim=(0, 2))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=time.device)
            / max(half - 1, 1)
        )
        values = time[:, None] * frequencies[None, :]
        output = torch.cat([values.sin(), values.cos()], dim=1)
        if output.size(1) < self.dim:
            output = F.pad(output, (0, 1))
        return output


class DiffCSP(nn.Module):
    """Small joint lattice/coordinate diffusion wrapper."""

    def __init__(
        self,
        denoiser,
        timesteps=1000,
        sigma_begin=0.005,
        sigma_end=0.5,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        self.time_embedding = SinusoidalTimeEmbedding(denoiser.time_dim)

        betas = cosine_beta_schedule(timesteps)
        betas = torch.cat([torch.zeros(1), betas])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        lattice_sigmas = torch.zeros_like(betas)
        lattice_sigmas[1:] = (
            betas[1:]
            * (1 - alphas_cumprod[:-1])
            / (1 - alphas_cumprod[1:])
        )

        coord_sigmas = torch.exp(
            torch.linspace(
                math.log(sigma_begin),
                math.log(sigma_end),
                timesteps,
            )
        )
        coord_sigmas = torch.cat([torch.zeros(1), coord_sigmas])
        coord_sigmas_norm = torch.cat(
            [torch.ones(1), estimate_score_variance(coord_sigmas[1:])]
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("lattice_sigmas", lattice_sigmas.sqrt())
        self.register_buffer("coord_sigmas", coord_sigmas)
        self.register_buffer("coord_sigmas_norm", coord_sigmas_norm)

    def forward(self, batch):
        batch_size = batch.num_graphs
        times = torch.randint(
            1,
            self.timesteps + 1,
            (batch_size,),
            device=batch.frac_coords.device,
        )
        time_emb = self.time_embedding(times)

        alpha_bar = self.alphas_cumprod[times]
        c0 = alpha_bar.sqrt()
        c1 = (1.0 - alpha_bar).sqrt()

        lattice = batch.lattice
        frac_coords = batch.frac_coords
        rand_lattice = torch.randn_like(lattice)
        rand_coords = torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattice + c1[:, None, None] * rand_lattice
        sigma = self.coord_sigmas[times].repeat_interleave(batch.num_atoms)[:, None]
        sigma_norm = self.coord_sigmas_norm[times].repeat_interleave(batch.num_atoms)[:, None]
        input_coords = (frac_coords + sigma * rand_coords) % 1.0

        pred_lattice, pred_coords = self.denoiser(
            time_emb,
            batch.atom_types,
            input_coords,
            input_lattice,
            batch.num_atoms,
            batch.batch,
        )

        target_coords = wrapped_normal_score(sigma * rand_coords, sigma) / sigma_norm.sqrt()
        loss_lattice = F.mse_loss(pred_lattice, rand_lattice)
        loss_coords = F.mse_loss(pred_coords, target_coords)
        return {
            "loss": loss_lattice + loss_coords,
            "loss_lattice": loss_lattice,
            "loss_coords": loss_coords,
        }

    @torch.no_grad()
    def sample(self, batch, step_lr=1e-5, progress=True):
        device = batch.atom_types.device
        lattice = torch.randn(
            batch.num_graphs,
            3,
            3,
            device=device,
        )
        frac_coords = torch.rand(batch.num_nodes, 3, device=device)

        iterator = range(self.timesteps, 0, -1)
        if progress:
            iterator = tqdm(iterator, desc="sample")

        for timestep in iterator:
            times = torch.full(
                (batch.num_graphs,),
                timestep,
                device=device,
                dtype=torch.long,
            )
            time_emb = self.time_embedding(times)
            alpha = self.alphas[timestep]
            alpha_bar = self.alphas_cumprod[timestep]
            lattice_sigma = self.lattice_sigmas[timestep]
            coord_sigma = self.coord_sigmas[timestep]
            coord_norm = self.coord_sigmas_norm[timestep]

            # Corrector: add a small amount of coordinate noise, then denoise.
            coord_noise = torch.randn_like(frac_coords) if timestep > 1 else 0
            corrector_step = step_lr * (
                coord_sigma / self.coord_sigmas[1]
            ).square()
            corrector_std = torch.sqrt(2 * corrector_step)

            _, pred_coords = self.denoiser(
                time_emb,
                batch.atom_types,
                frac_coords,
                lattice,
                batch.num_atoms,
                batch.batch,
            )
            half_coords = (
                frac_coords
                - corrector_step * pred_coords * coord_norm.sqrt()
                + corrector_std * coord_noise
            )

            # Predictor: move one diffusion step toward t-1.
            coord_sigma_prev = self.coord_sigmas[timestep - 1]
            predictor_step = coord_sigma.square() - coord_sigma_prev.square()
            predictor_std = (
                coord_sigma_prev.square()
                * predictor_step
                / coord_sigma.square()
            ).sqrt()

            # Fresh noise for the predictor; the corrector and predictor must
            # not share noise.
            lattice_noise = torch.randn_like(lattice) if timestep > 1 else 0
            coord_noise = torch.randn_like(frac_coords) if timestep > 1 else 0

            pred_lattice, pred_coords = self.denoiser(
                time_emb,
                batch.atom_types,
                half_coords,
                lattice,
                batch.num_atoms,
                batch.batch,
            )

            c0 = alpha.rsqrt()
            c1 = (1 - alpha) / (1 - alpha_bar).sqrt()
            lattice = c0 * (lattice - c1 * pred_lattice)
            lattice = lattice + lattice_sigma * lattice_noise

            frac_coords = (
                half_coords
                - predictor_step * pred_coords * coord_norm.sqrt()
                + predictor_std * coord_noise
            )
            frac_coords = frac_coords % 1.0

        return {
            "lattice": lattice,
            "frac_coords": frac_coords,
            "atom_types": batch.atom_types,
            "num_atoms": batch.num_atoms,
        }
