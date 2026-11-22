"""Train a compact DiffCSP model without Hydra or Lightning."""

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data import CrystalDataset, make_loader
from .diffusion import DiffCSP
from .model1 import CrysformerConfig
from .model2 import CrystalDualStreamConfig
from .utils import build_denoiser, resolve_device, seed_everything, torch_load


def parse_args():
    parser = argparse.ArgumentParser("Train a compact DiffCSP")
    parser.add_argument("--data-dir", default="data/carbon_24")
    parser.add_argument(
        "--model",
        default="crysformer",
        choices=["crysformer", "dualstream", "cspnet"],
    )
    parser.add_argument("--output-dir", default="outputs/diffcsp_simple")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-frequencies", type=int, default=128)
    parser.add_argument(
        "--sparse-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dualstream: only build edges for real atom pairs (default) "
        "instead of the dense padded grid; use --no-sparse-edges for A/B.",
    )
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume")
    return parser.parse_args()


def move_batch(batch, device):
    return batch.to(device)


def run_epoch(diffusion, loader, optimizer, device, training=True):
    diffusion.train(training)
    total_loss = 0.0
    total_graphs = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            losses = diffusion(batch)
            loss = losses["loss"]

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 0.5)
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


def save_checkpoint(path, diffusion, optimizer, scheduler, args, epoch, best_val, model_config):
    torch.save(
        {
            "diffusion": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,
            "timesteps": args.timesteps,
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_dataset = CrystalDataset(
        data_dir / "train.csv",
        limit=args.limit_train,
        rebuild_cache=args.rebuild_cache,
    )
    val_dataset = CrystalDataset(
        data_dir / "val.csv",
        limit=args.limit_val,
        rebuild_cache=args.rebuild_cache,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.model == "cspnet":
        config = {
            "hidden_dim": args.hidden_dim,
            "time_dim": args.time_dim,
            "num_layers": args.num_layers,
            "max_atoms": args.max_atoms,
            "num_frequencies": args.num_frequencies,
        }
    elif args.model == "dualstream":
        config = asdict(
            CrystalDualStreamConfig(
                hidden_dim=args.hidden_dim,
                time_dim=args.time_dim,
                num_layers=args.num_layers,
                max_atomic_number=args.max_atoms,
                sparse_edges=args.sparse_edges,
            )
        )
    else:
        config = asdict(
            CrysformerConfig(
                hidden_dim=args.hidden_dim,
                time_dim=args.time_dim,
                num_layers=args.num_layers,
                num_frequencies=args.num_frequencies,
                max_atomic_number=args.max_atoms,
                use_energy_head=False,
            )
        )
    model_config = {"model": args.model, "config": config}
    diffusion = DiffCSP(
        build_denoiser(args.model, config),
        timesteps=args.timesteps,
    ).to(device)
    optimizer = Adam(
        diffusion.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.6,
        patience=30,
        min_lr=1e-4,
    )

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        checkpoint = torch_load(args.resume, map_location=device)
        diffusion.load_state_dict(checkpoint["diffusion"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]

    metrics_path = output_dir / "metrics.csv"
    write_header = not metrics_path.exists() or args.resume is None
    metrics_file = metrics_path.open("a" if args.resume else "w", newline="")
    writer = csv.writer(metrics_file)
    if write_header:
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "seconds"])

    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_loss = run_epoch(diffusion, train_loader, optimizer, device, training=True)
        val_loss = run_epoch(diffusion, val_loader, optimizer, device, training=False)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        writer.writerow([epoch, train_loss, val_loss, lr, round(elapsed, 3)])
        metrics_file.flush()
        print(
            f"epoch {epoch:04d} | train {train_loss:.6f} | "
            f"val {val_loss:.6f} | lr {lr:.2e} | {elapsed:.1f}s"
        )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                diffusion,
                optimizer,
                scheduler,
                args,
                epoch,
                best_val,
                model_config,
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            output_dir / "last.pt",
            diffusion,
            optimizer,
            scheduler,
            args,
            epoch,
            best_val,
            model_config,
        )

        if stale_epochs >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    metrics_file.close()
    print(f"done | best val {best_val:.6f} | output {output_dir}")


if __name__ == "__main__":
    main()
