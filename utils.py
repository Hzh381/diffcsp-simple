"""Small checkpoint and structure helpers shared by scripts."""

from pathlib import Path

import torch
from chemparse import parse_formula
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter

from .diffusion import DiffCSP
from .model import CSPNet
from .model1 import CrysformerConfig, CrysformerDenoiser
from .model2 import CrystalDualStreamConfig, CrystalDualStreamDenoiser


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_denoiser(model_name, model_config):
    """Instantiate the selected denoiser from its serialized config."""
    if model_name == "cspnet":
        return CSPNet(**model_config)
    if model_name == "dualstream":
        return CrystalDualStreamDenoiser(CrystalDualStreamConfig(**model_config))
    return CrysformerDenoiser(CrysformerConfig(**model_config))


def load_diffusion(checkpoint_path, device):
    checkpoint = torch_load(checkpoint_path, map_location=device)
    model_info = checkpoint["model_config"]
    if "model" in model_info:
        model = build_denoiser(model_info["model"], model_info["config"])
    elif "max_atoms" in model_info:##此处做了修改
        # Flat cspnet config written before configs were tagged with a model name.
        model = build_denoiser("cspnet", model_info)###此处做了修改
    else:
        # Checkpoints written before the model switch stored a flat config.
        model = CrysformerDenoiser(CrysformerConfig(**model_info))
    diffusion = DiffCSP(model, timesteps=checkpoint["timesteps"])
    diffusion.load_state_dict(checkpoint["diffusion"])
    diffusion.to(device).eval()
    return diffusion, checkpoint


def formula_to_atom_types(formula):
    composition = parse_formula(formula)
    atom_types = []
    for symbol, count in composition.items():
        atom_types.extend([Element(symbol).Z] * int(round(count)))
    if not atom_types:
        raise ValueError(f"Invalid formula: {formula}")
    return torch.tensor(atom_types, dtype=torch.long)


def structure_from_arrays(lattice, frac_coords, atom_types):
    return Structure(
        lattice=Lattice(lattice.detach().cpu().numpy()),
        species=atom_types.detach().cpu().tolist(),
        coords=frac_coords.detach().cpu().numpy(),
        coords_are_cartesian=False,
    ).get_sorted_structure()


def write_cif(structure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    CifWriter(structure).write_file(str(path))
