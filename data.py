"""Crystal data loading with a small, readable cache format."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pymatgen.core import Structure


CACHE_VERSION = 1


def _torch_load(path):
    """Load old and new PyTorch checkpoints/cache files consistently."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_crystal(cif):
    """Convert one CIF string into the three tensors used by the model."""
    structure = Structure.from_str(cif, fmt="cif").get_reduced_structure()
    return {
        "atom_types": torch.tensor(
            [site.specie.Z for site in structure.sites],
            dtype=torch.long,
        ),
        "frac_coords": torch.tensor(
            structure.frac_coords % 1.0,
            dtype=torch.float32,
        ),
        "lattice": torch.tensor(
            structure.lattice.matrix,
            dtype=torch.float32,
        ),
    }


class CrystalBatch:
    """Minimal batch object: concatenated atoms plus graph boundaries."""

    def __init__(self, atom_types, frac_coords, lattice, num_atoms, batch):
        self.atom_types = atom_types
        self.frac_coords = frac_coords
        self.lattice = lattice
        self.num_atoms = num_atoms
        self.batch = batch
        self.num_graphs = len(num_atoms)
        self.num_nodes = len(atom_types)

    @classmethod
    def from_samples(cls, atom_types_list):
        num_atoms = torch.tensor(
            [len(atom_types) for atom_types in atom_types_list],
            dtype=torch.long,
        )
        atom_types = torch.cat(atom_types_list)
        batch = torch.repeat_interleave(
            torch.arange(len(num_atoms)),
            num_atoms,
        )
        return cls(atom_types, None, None, num_atoms, batch)

    def to(self, device):
        return CrystalBatch(
            self.atom_types.to(device),
            self.frac_coords.to(device) if self.frac_coords is not None else None,
            self.lattice.to(device) if self.lattice is not None else None,
            self.num_atoms.to(device),
            self.batch.to(device),
        )

    def to_dicts(self):
        records = []
        offset = 0
        for lattice, num_atom in zip(self.lattice, self.num_atoms.tolist()):
            end = offset + num_atom
            records.append(
                {
                    "lattice": lattice,
                    "frac_coords": self.frac_coords[offset:end],
                    "atom_types": self.atom_types[offset:end],
                    "num_atoms": num_atom,
                }
            )
            offset = end
        return records


class CrystalDataset(Dataset):
    """Read CSV, cache parsed CIFs, and return PyG Data objects."""

    def __init__(
        self,
        csv_path,
        cache_path=None,
        limit=None,
        rebuild_cache=False,
    ):
        self.csv_path = Path(csv_path)
        self.cache_path = Path(cache_path) if cache_path else (
            self.csv_path.parent / f".simple_{self.csv_path.stem}.pt"
        )
        self.records = self._load_records(limit, rebuild_cache)

    def _load_records(self, limit, rebuild_cache):
        if self.cache_path.exists() and not rebuild_cache:
            payload = _torch_load(self.cache_path)
            if payload.get("version") == CACHE_VERSION:
                records = payload["records"]
                return records[:limit] if limit else records

        frame = pd.read_csv(self.csv_path)
        if limit:
            frame = frame.iloc[:limit]

        records = []
        for row in tqdm(frame.itertuples(index=False), desc=f"parse {self.csv_path.name}"):
            records.append(parse_crystal(row.cif))

        # Do not replace a full cache with a temporary --limit subset.
        if limit:
            return records

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"version": CACHE_VERSION, "records": records},
            self.cache_path,
        )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return record


def collate_graphs(data_list):
    """Concatenate variable-size crystals into one batch."""
    num_atoms = torch.tensor(
        [len(record["atom_types"]) for record in data_list],
        dtype=torch.long,
    )
    atom_types = torch.cat([record["atom_types"] for record in data_list])
    frac_coords = torch.cat([record["frac_coords"] for record in data_list])
    lattice = torch.stack([record["lattice"] for record in data_list])
    batch = torch.repeat_interleave(
        torch.arange(len(num_atoms)),
        num_atoms,
    )
    return CrystalBatch(atom_types, frac_coords, lattice, num_atoms, batch)


def make_loader(dataset, batch_size, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )
