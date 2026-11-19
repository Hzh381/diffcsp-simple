"""Dependency-light implementation of Crysformer (DAO, Figure 5).

Shapes use ``B`` batches and ``N`` atoms. ``lattice`` stores basis vectors as
rows, so Cartesian coordinates are ``fractional @ lattice``. Dense all-pairs
attention keeps the code portable; it requires only PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class CrysformerConfig:
    """DAO main-model defaults, with an optional energy head."""

    hidden_dim: int = 384
    num_layers: int = 12
    heads: int = 8
    num_frequencies: int = 128
    ffn_multiplier: int = 4
    time_dim: int = 128
    max_atomic_number: int = 100
    dropout: float = 0.0
    use_energy_head: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.time_dim % 2:
            raise ValueError("time_dim must be even")


@dataclass
class CrysformerOutput:
    """Outputs from Figure 5e, plus the learned node/graph features."""

    frac_score: Tensor
    lattice_noise: Tensor
    energy: Optional[Tensor]
    node_features: Tensor
    graph_features: Tensor


class AtomEncoder(nn.Module):
    """Learned atomic embedding, or an optional fixed CGCNN feature table."""

    def __init__(
        self,
        hidden_dim: int,
        max_atomic_number: int,
        cgcnn_features: Optional[Tensor],
    ):
        super().__init__()
        self.embedding: Optional[nn.Embedding] = None
        self.projection: Optional[nn.Linear] = None
        self.register_buffer("table", None)

        if cgcnn_features is None:
            self.embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        else:
            table = torch.as_tensor(cgcnn_features, dtype=torch.get_default_dtype())
            if table.ndim != 2 or table.shape[0] < max_atomic_number + 1:
                raise ValueError("invalid CGCNN feature-table shape")
            self.table = table
            self.projection = nn.Linear(table.shape[1], hidden_dim)

    def forward(self, atom_types: Tensor) -> Tensor:
        if self.embedding is not None:
            return self.embedding(atom_types)
        return self.projection(self.table[atom_types])


class PairEmbedding(nn.Module):
    """Invariant edge ``e_ij``: lattice Gramian plus Fourier displacements."""

    def __init__(self, num_frequencies: int):
        super().__init__()
        frequencies = 2.0 * math.pi * torch.arange(num_frequencies)
        self.register_buffer("frequencies", frequencies)
        self.edge_dim = 9 + 6 * num_frequencies
        self.norm = nn.LayerNorm(self.edge_dim)

    def forward(self, frac: Tensor, lattice: Tensor) -> Tensor:
        # Element [i, j] is the periodic displacement F_j - F_i.
        displacement = (frac[:, None, :, :] - frac[:, :, None, :]) % 1.0
        phase = displacement.unsqueeze(-1) * self.frequencies.to(displacement)
        fourier = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)

        batch, atoms, _, _ = displacement.shape
        gramian = (lattice @ lattice.transpose(-1, -2)).reshape(batch, 1, 1, 9)
        gramian = gramian.expand(batch, atoms, atoms, 9)
        return self.norm(torch.cat((gramian, fourier), dim=-1))


class GraphAttention(nn.Module):
    """Multi-head attention over invariant node and edge features."""

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge = nn.Linear(edge_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node: Tensor, edge: Tensor, mask: Tensor) -> Tensor:
        batch, atoms, hidden = node.shape
        heads, head_dim = self.heads, self.head_dim

        query = self.query(node).view(batch, atoms, heads, head_dim).transpose(1, 2)
        key = self.key(node).view(batch, atoms, heads, head_dim).transpose(1, 2)
        value = self.value(node).view(batch, atoms, heads, head_dim).transpose(1, 2)
        edge = (
            self.edge(edge)
            .view(batch, atoms, atoms, heads, head_dim)
            .permute(0, 3, 1, 2, 4)
        )

        key = key[:, :, None, :, :] + edge
        value = value[:, :, None, :, :] + edge
        logits = torch.einsum("bhid,bhijd->bhij", query, key)
        logits = logits / math.sqrt(head_dim)
        logits = logits.masked_fill(
            ~mask[:, None, None, :], torch.finfo(logits.dtype).min
        )

        attention = self.dropout(logits.softmax(dim=-1))
        message = torch.einsum("bhij,bhijd->bhid", attention, value)
        message = message.transpose(1, 2).reshape(batch, atoms, hidden)
        return self.output(message) * mask[..., None].to(message.dtype)


class GatedResidual(nn.Module):
    """Learn ``alpha`` in ``alpha * x + (1-alpha) * update``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(3 * hidden_dim, 1, bias=False)

    def forward(self, value: Tensor, update: Tensor) -> Tensor:
        alpha = self.gate(torch.cat((value, update, value - update), -1)).sigmoid()
        return alpha * value + (1.0 - alpha) * update


class CrysformerBlock(nn.Module):
    """PreNorm attention and feed-forward blocks with gated residuals."""

    def __init__(self, config: CrysformerConfig, edge_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = GraphAttention(
            config.hidden_dim, edge_dim, config.heads, config.dropout
        )
        self.gate1 = GatedResidual(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_multiplier * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_multiplier * config.hidden_dim, config.hidden_dim),
        )
        self.gate2 = GatedResidual(config.hidden_dim)

    def forward(self, node: Tensor, edge: Tensor, mask: Tensor) -> Tensor:
        node = self.gate1(node, self.attention(self.norm1(node), edge, mask))
        return self.gate2(node, self.mlp(self.norm2(node)))


class Crysformer(nn.Module):
    """Crysformer network for DAO-G and DAO-P.

    ``atom_types``: ``[B, N]``; ``frac_coords``: ``[B, N, 3]``;
    ``lattice``: ``[B, 3, 3]``; ``time_features``: optional ``[B, time_dim]``
    sinusoidal embedding supplied by the caller; ``node_mask``: optional
    Boolean ``[B, N]``.
    """

    def __init__(
        self,
        config: Optional[CrysformerConfig] = None,
        cgcnn_features: Optional[Tensor] = None,
    ):
        super().__init__()
        self.config = config or CrysformerConfig()
        config = self.config

        self.atom = AtomEncoder(
            config.hidden_dim, config.max_atomic_number, cgcnn_features
        )
        self.node = nn.Linear(config.hidden_dim + config.time_dim, config.hidden_dim)
        self.pair = PairEmbedding(config.num_frequencies)
        self.blocks = nn.ModuleList(
            CrysformerBlock(config, self.pair.edge_dim)
            for _ in range(config.num_layers)
        )

        self.frac_head = nn.Linear(config.hidden_dim, 3, bias=False)
        self.lattice_head = nn.Linear(config.hidden_dim, 9, bias=False)
        self.energy_head = (
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )
            if config.use_energy_head
            else None
        )

    def forward(
        self,
        atom_types: Tensor,
        frac_coords: Tensor,
        lattice: Tensor,
        time_features: Optional[Tensor] = None,
        node_mask: Optional[Tensor] = None,
    ) -> CrysformerOutput:
        batch, atoms = atom_types.shape
        if frac_coords.shape != (batch, atoms, 3) or lattice.shape != (batch, 3, 3):
            raise ValueError("expected frac [B,N,3] and lattice [B,3,3]")

        atom_types = atom_types.to(lattice.device).long()
        frac_coords = frac_coords.to(lattice)
        mask = (
            torch.ones((batch, atoms), dtype=torch.bool, device=lattice.device)
            if node_mask is None
            else node_mask.to(lattice.device, dtype=torch.bool)
        )
        if time_features is None:
            time_features = frac_coords.new_zeros((batch, self.config.time_dim))
        else:
            time_features = time_features.to(frac_coords)

        node = self.atom(atom_types)
        node = torch.cat((node, time_features[:, None].expand(-1, atoms, -1)), -1)
        node = self.node(node) * mask[..., None].to(lattice)

        edge = self.pair(frac_coords, lattice)
        for block in self.blocks:
            node = block(node, edge, mask)

        count = mask.sum(1, keepdim=True).clamp_min(1)
        graph = (node * mask[..., None]).sum(1) / count.to(node)
        frac_score = self.frac_head(node)
        lattice_noise = self.lattice_head(graph).reshape(batch, 3, 3) @ lattice
        energy = self.energy_head(graph).squeeze(-1) if self.energy_head else None
        return CrysformerOutput(
            frac_score, lattice_noise, energy, node, graph
        )


class CrysformerDenoiser(nn.Module):
    """Adapt :class:`Crysformer` to the flat-batch denoiser interface.

    ``diffusion.DiffCSP`` stores a batch as concatenated atoms plus graph
    boundaries and supplies one sinusoidal time embedding per graph. This
    wrapper pads atoms into ``[B, N]``, builds the node mask, and flattens the
    per-atom score back to the original concatenated order.
    """

    def __init__(self, config: Optional[CrysformerConfig] = None):
        super().__init__()
        self.config = config or CrysformerConfig()
        self.net = Crysformer(self.config)
        self.time_dim = self.config.time_dim

    def forward(
        self,
        time_emb: Tensor,
        atom_types: Tensor,
        frac_coords: Tensor,
        lattice: Tensor,
        num_atoms: Tensor,
        node2graph: Tensor,
    ) -> tuple:
        batch = lattice.size(0)
        max_atoms = int(num_atoms.max())
        device = atom_types.device

        offsets = torch.cumsum(num_atoms, 0) - num_atoms
        positions = torch.arange(atom_types.size(0), device=device) - offsets[node2graph]

        padded_types = atom_types.new_zeros((batch, max_atoms))
        padded_frac = frac_coords.new_zeros((batch, max_atoms, 3))
        node_mask = torch.zeros((batch, max_atoms), dtype=torch.bool, device=device)
        padded_types[node2graph, positions] = atom_types
        padded_frac[node2graph, positions] = frac_coords
        node_mask[node2graph, positions] = True

        output = self.net(
            padded_types,
            padded_frac,
            lattice,
            time_features=time_emb,
            node_mask=node_mask,
        )
        return output.lattice_noise, output.frac_score[node_mask]


__all__ = [
    "Crysformer",
    "CrysformerConfig",
    "CrysformerDenoiser",
    "CrysformerOutput",
]
