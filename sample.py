"""Generate CIF files for an arbitrary chemical formula."""

import argparse
from pathlib import Path

from .data import CrystalBatch
from .utils import (
    formula_to_atom_types,
    load_diffusion,
    resolve_device,
    seed_everything,
    structure_from_arrays,
    write_cif,
)


def parse_args():
    parser = argparse.ArgumentParser("Sample crystals from a compact DiffCSP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/diffcsp_simple/samples")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-lr", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    diffusion, _ = load_diffusion(args.checkpoint, device)
    atom_types = formula_to_atom_types(args.formula)
    output_dir = Path(args.output_dir) / args.formula
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    remaining = args.num_samples
    while remaining > 0:
        batch_size = min(args.batch_size, remaining)
        atom_types_list = [
            atom_types.clone()
            for _ in range(batch_size)
        ]
        batch = CrystalBatch.from_samples(atom_types_list).to(device)
        generated = diffusion.sample(batch, step_lr=args.step_lr)

        offset = 0
        for index in range(batch_size):
            num_atom = len(atom_types)
            structure = structure_from_arrays(
                generated["lattice"][index],
                generated["frac_coords"][offset : offset + num_atom],
                generated["atom_types"][offset : offset + num_atom],
            )
            write_cif(structure, output_dir / f"{args.formula}_{written + 1:04d}.cif")
            written += 1
            offset += num_atom
        remaining -= batch_size

    print(f"wrote {written} CIF files to {output_dir}")


if __name__ == "__main__":
    main()
