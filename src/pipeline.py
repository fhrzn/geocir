"""
End-to-end training pipeline: train → index → eval.

Usage (full run):
    python -m src.pipeline \
        --run-name my-run \
        --train-data /path/to/merged_set_train.csv \
        --index-data /path/to/merged_set_index.csv \
        --img-base-path /mnt/yokoyamalab-nas/gldv2-full

Skip stages (if already done):
    python -m src.pipeline \
        --run-name my-run \
        --train-data ... --index-data ... --img-base-path ... \
        --skip-train --ckpt-path checkpoints/my-run/best.pt
        --skip-index --index-dir index/my-run
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from src.evaluate import run_eval
from src.geotir.train import train
from src.index_geotir import ingest


def _banner(msg: str):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


def run_pipeline(args):
    run_name = args.run_name
    ckpt_dir = Path("checkpoints") / run_name
    index_dir = Path("index") / run_name
    results_path = Path("eval") / f"{run_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 1: Train                                                       #
    # ------------------------------------------------------------------ #
    ckpt_path = args.ckpt_path or str(ckpt_dir / "best.pt")

    if args.skip_train:
        _banner(f"[1/3] Train — SKIPPED (using checkpoint: {ckpt_path})")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        _banner(f"[1/3] Train — run: {run_name}")
        train_args = SimpleNamespace(
            data=args.train_data,
            img_path=args.img_base_path,
            run_name=run_name,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            pair_cap=args.pair_cap,
            warmup_ratio=args.warmup_ratio,
        )
        train(train_args)

    # ------------------------------------------------------------------ #
    # Stage 2: Index                                                       #
    # ------------------------------------------------------------------ #
    resolved_index_dir = args.index_dir or str(index_dir)

    if args.skip_index:
        _banner(f"[2/3] Index — SKIPPED (using index: {resolved_index_dir})")
        if not (Path(resolved_index_dir) / "index.index").exists():
            raise FileNotFoundError(f"Index not found at: {resolved_index_dir}")
    else:
        _banner(f"[2/3] Index — output: {resolved_index_dir}")
        index_args = SimpleNamespace(
            data_path=args.index_data,
            ckpt_path=ckpt_path,
            img_base_path=args.img_base_path,
            img_col="id",
            id_col="id",
            batch_size=args.index_batch_size,
            index_size=768,
            index_type=args.index_type,
            output_dir=resolved_index_dir,
        )
        ingest(index_args)

    # ------------------------------------------------------------------ #
    # Stage 3: Eval                                                        #
    # ------------------------------------------------------------------ #
    _banner(f"[3/3] Eval — results → {results_path}")
    eval_args = SimpleNamespace(
        index_dir=resolved_index_dir,
        ckpt_path=ckpt_path,
        min_relevant=args.min_relevant,
        batch_size=args.eval_batch_size,
        breakdown=args.breakdown,
        output=str(results_path),
        device=args.device,
    )
    run_eval(eval_args)

    _banner("Pipeline complete")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Index      : {resolved_index_dir}")
    print(f"  Results    : {results_path}")

    with open(results_path) as f:
        results = json.load(f)
    print("\nOverall metrics:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train → Index → Eval pipeline for GeoTIR"
    )

    # Shared
    parser.add_argument("--run-name", required=True, help="Name for this run (used for checkpoint/index/eval paths)")
    parser.add_argument("--img-base-path", required=True, help="Base image directory (parent of train/index subfolders)")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g. cuda, cpu). Auto-detected if omitted.")

    # Stage skipping
    parser.add_argument("--skip-train", action="store_true", help="Skip training; requires --ckpt-path")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing; requires --index-dir")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Checkpoint to use when skipping train (default: checkpoints/<run-name>/best.pt)")
    parser.add_argument("--index-dir", type=str, default=None, help="Index directory to use when skipping index (default: index/<run-name>)")

    # Data
    train_grp = parser.add_argument_group("data")
    train_grp.add_argument("--train-data", required=True, help="CSV for training")
    train_grp.add_argument("--index-data", required=True, help="CSV for indexing")

    # Train hyperparams
    tr = parser.add_argument_group("train")
    tr.add_argument("--epochs", type=int, default=5)
    tr.add_argument("--train-batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight-decay", type=float, default=0.01)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=32)
    tr.add_argument("--lora-dropout", type=float, default=0.05)
    tr.add_argument("--pair-cap", type=int, default=None, help="Auto-computed from data if omitted")
    tr.add_argument("--warmup-ratio", type=float, default=0.05)

    # Index params
    ix = parser.add_argument_group("index")
    ix.add_argument("--index-batch-size", type=int, default=512)
    ix.add_argument("--index-type", default="flat_ip")

    # Eval params
    ev = parser.add_argument_group("eval")
    ev.add_argument("--min-relevant", type=int, default=5)
    ev.add_argument("--eval-batch-size", type=int, default=256)
    ev.add_argument("--breakdown", action="store_true", help="Per-category and per-country mAP breakdown")

    args = parser.parse_args()

    if args.skip_train and not args.ckpt_path:
        # Fall back to default path; existence is checked inside run_pipeline
        pass
    if args.skip_index and not args.index_dir:
        pass

    run_pipeline(args)


if __name__ == "__main__":
    main()
