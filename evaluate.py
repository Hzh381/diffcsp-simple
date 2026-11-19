"""Reconstruct the test split and report structure-matching metrics."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import smact
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from smact.screening import pauling_test
from torch.utils.data import Subset

from .data import CrystalDataset, make_loader
from .utils import load_diffusion, resolve_device, seed_everything, structure_from_arrays


STEP_LR = {
    "perov_5": 5e-7,
    "carbon_24": 5e-6,
    "mp_20": 1e-5,
    "mpts_52": 1e-5,
}


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a compact DiffCSP checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/carbon_24")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--step-lr", type=float, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    return parser.parse_args()


def structure_match(matcher, predicted, target):
    try:
        result = matcher.get_rms_dist(predicted, target)
        return None if result is None else result[0]
    except Exception:
        return None


def structure_is_valid(structure):
    try:
        distances = structure.distance_matrix
        distances = distances + np.eye(len(structure)) * 1e6
        return distances.min() >= 0.5 and structure.volume >= 0.1
    except Exception:
        return False


def composition_is_valid(structure):
    symbols = [site.specie.symbol for site in structure.sites]
    if len(set(symbols)) == 1:
        return True
    if all(symbol in smact.metals for symbol in symbols):
        return True

    counter = Counter(symbols)
    elements = tuple(counter.keys())
    counts = tuple(counter.values())
    space = smact.element_dictionary(elements)
    if any(element is None for element in space.values()):
        return False
    electronegativities = [element.pauling_eneg for element in space.values()]
    oxidation_states = [element.oxidation_states for element in space.values()]
    threshold = max(counts)

    for states in itertools.product(*oxidation_states):
        neutral, _ = smact.neutral_ratios(
            states,
            stoichs=[(count,) for count in counts],
            threshold=threshold,
        )
        if neutral:
            try:
                if pauling_test(states, electronegativities):
                    return True
            except TypeError:
                return True
    return False


def crystal_is_valid(structure):
    return structure_is_valid(structure) and composition_is_valid(structure)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    diffusion, _ = load_diffusion(args.checkpoint, device)

    data_dir = Path(args.data_dir)
    dataset = CrystalDataset(data_dir / f"{args.split}.csv")
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = make_loader(dataset, args.batch_size, shuffle=False)

    matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
    matched = 0
    total = 0
    rms_values = []
    step_lr = args.step_lr
    if step_lr < 0:
        step_lr = STEP_LR.get(data_dir.name, 1e-5)

    for batch in loader:
        ground_truth = batch.to_dicts()
        batch = batch.to(device)
        generated_samples = [
            diffusion.sample(batch, step_lr=step_lr, progress=False)
            for _ in range(args.num_samples)
        ]

        offset = 0
        for index, target_data in enumerate(ground_truth):
            num_atom = int(target_data["num_atoms"])
            target = structure_from_arrays(
                target_data["lattice"],
                target_data["frac_coords"],
                target_data["atom_types"],
            )
            total += 1
            if not crystal_is_valid(target):
                offset += num_atom
                continue

            best_rms = None
            for generated in generated_samples:
                predicted = structure_from_arrays(
                    generated["lattice"][index],
                    generated["frac_coords"][offset : offset + num_atom],
                    generated["atom_types"][offset : offset + num_atom],
                )
                if not crystal_is_valid(predicted):
                    continue
                rms = structure_match(matcher, predicted, target)
                if rms is not None:
                    best_rms = rms if best_rms is None else min(best_rms, rms)

            if best_rms is not None:
                matched += 1
                rms_values.append(float(best_rms))
            offset += num_atom

        print(f"evaluated {total} structures")

    metrics = {
        "matched": matched,
        "total": total,
        "match_rate": matched / max(total, 1),
        "mean_rmsd": sum(rms_values) / len(rms_values) if rms_values else None,
    }
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "eval_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
