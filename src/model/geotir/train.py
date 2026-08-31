import argparse
from pathlib import Path

import polars as pl
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from src.data.data import GeoTIRDataset, geo_collate_fn, warm_image_cache
from src.model.geotir.model import GeoTIRModel
from src.utils import get_device

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
WANDB_PROJECT = "geotir-clip-finetune"


def read_split(path: str, sample_n: int = 0, seed: int = 42) -> pl.DataFrame:
    df = pl.read_csv(path)
    # older exports named the label column "pred_label"; current data has "category"
    if "category" not in df.columns and "pred_label" in df.columns:
        df = df.rename({"pred_label": "category"})
    if sample_n and sample_n < df.height:
        df = df.sample(n=sample_n, seed=seed, shuffle=True)
        print(f"sampled {df.height} rows from {path}")
    return df


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc="validate", leave=False)
        for batch in pbar:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(batch)
            total_loss += output["loss"].item()
            pbar.set_postfix(loss=f"{output['loss'].item():.4f}")
    return total_loss / len(loader)


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, global_step):
    model.train()
    trloss = 0.0
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)

    for batch in pbar:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model(batch)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        trloss += loss.item()
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        wandb.log(
            {
                "train/loss": loss.item(),
                "train/temperature": output["temperature"],
                "train/lr": scheduler.get_last_lr()[0],
            },
            step=global_step,
        )

    return trloss / len(loader), global_step


def train(args):
    device = get_device()

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    train_df = read_split(args.data, args.sample_n, args.seed)

    # Auto-compute pair_cap from data if not explicitly set:
    # 75th-percentile of (category, country) group sizes // 2, clamped to at least 1.
    # if args.pair_cap is None:
    #     group_sizes = (
    #         train_df.group_by(["category", "country"])
    #         .agg(pl.len().alias("n"))
    #         .filter(pl.col("n") >= 2)["n"]
    #     )
    #     args.pair_cap = max(int(group_sizes.quantile(0.75)) // 2, 1)
    #     print(f"Auto pair_cap: {args.pair_cap}")

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=geo_collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    dataset = GeoTIRDataset(
        df=train_df,
        base_img_path=args.img_path,
        processor=processor,
        use_template=args.use_caption_template,
        max_text_length=args.max_text_length,
        src_col=args.src_col,
        caption_col=args.caption_col,
        cache_dir=args.img_cache_dir,
        cache_size=args.cache_size,
    )
    if args.warm_cache_workers > 0:
        warm_image_cache(dataset, workers=args.warm_cache_workers)
    loader = DataLoader(dataset, **dl_kwargs)

    val_loader = None
    if args.val_data:
        val_df = read_split(args.val_data, args.sample_n, args.seed)
        val_dataset = GeoTIRDataset(
            df=val_df,
            base_img_path=args.img_path,
            processor=processor,
            use_template=args.use_caption_template,
            max_text_length=args.max_text_length,
            src_col=args.src_col,
            caption_col=args.caption_col,
            cache_dir=args.img_cache_dir,
            cache_size=args.cache_size,
        )
        if args.warm_cache_workers > 0:
            warm_image_cache(val_dataset, workers=args.warm_cache_workers)
        val_loader = DataLoader(val_dataset, **dl_kwargs)

    model = GeoTIRModel(
        clip_model_name=CLIP_MODEL_NAME,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # total_steps = len(sampler) * args.epochs
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    hparams = {
        "model": CLIP_MODEL_NAME,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": "q_proj,v_proj",
        "batch_size": args.batch_size,
        "max_text_length": args.max_text_length,
        "use_caption_template": args.use_caption_template,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        # "pair_cap": args.pair_cap,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": "cosine",
    }

    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    epochs_no_improve = 0
    with wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=hparams,
        mode=args.wandb_mode,
    ) as run:
        print(f"W&B run: {run.url}")
        print(f"Checkpoints: {ckpt_dir}")

        for epoch in tqdm(range(1, args.epochs + 1), desc="finetune"):
            epoch_loss, global_step = train_one_epoch(
                model, loader, optimizer, scheduler, device, epoch, global_step
            )
            wandb.log(
                {"epoch/train_loss": epoch_loss, "epoch": epoch}, step=global_step
            )
            tqdm.write(f"epoch {epoch} | train_loss {epoch_loss:.4f}")

            monitor_loss = epoch_loss
            if val_loader is not None:
                val_loss = validate(model, val_loader, device)
                wandb.log(
                    {"epoch/val_loss": val_loss, "epoch": epoch}, step=global_step
                )
                tqdm.write(f"epoch {epoch} | val_loss   {val_loss:.4f}")
                monitor_loss = val_loss

            improved = monitor_loss < best_loss - args.min_delta
            if improved:
                best_loss = monitor_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if not args.no_save:
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": epoch_loss,
                }
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:02d}.pt")

                if improved:
                    best_path = ckpt_dir / "best.pt"
                    torch.save(ckpt, best_path)
                    artifact = wandb.Artifact(
                        name=f"{args.run_name}-best",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "train_loss": epoch_loss,
                            "monitor_loss": best_loss,
                        },
                    )
                    artifact.add_file(str(best_path))
                    run.log_artifact(artifact)
                    tqdm.write(f"  -> best saved (loss {best_loss:.4f})")

            if (
                args.early_stop_patience > 0
                and epochs_no_improve >= args.early_stop_patience
            ):
                tqdm.write(
                    f"early stop: no improvement for {epochs_no_improve} epoch(s) "
                    f"(best {best_loss:.4f})"
                )
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--run-name", default="clip-vit-large-lora-r16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--use-caption-template", action="store_true")
    parser.add_argument("--src-col", default="src")
    parser.add_argument("--caption-col", default="caption")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches prefetched per worker")
    parser.add_argument("--img-cache-dir", default=None,
                        help="local dir for downscaled JPEGs; avoids re-reading the NAS "
                             "every epoch (set to a fast local disk)")
    parser.add_argument("--cache-size", type=int, default=256,
                        help="longer-side pixel size for cached JPEGs")
    parser.add_argument("--warm-cache-workers", type=int, default=0,
                        help="threads to pre-fill the cache from the NAS before training "
                             "(0 = fill lazily during epoch 1)")
    parser.add_argument("--sample-n", type=int, default=0,
                        help="randomly sample N rows from train/val (0 = use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing checkpoints / artifacts (smoke tests)")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="stop after N epochs with no monitored-loss improvement "
                             "(0 = disabled)")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="minimum decrease in monitored loss to count as improvement")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--max-text-length", type=int, default=77)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # parser.add_argument(
    #     "--pair-cap",
    #     type=int,
    #     default=None,
    #     help="Max pairs per (category, country) group per epoch. Auto-computed from data if omitted.",
    # )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear warmup",
    )
    parser.add_argument(
        "--wandb-project", default=WANDB_PROJECT, help="W&B project name"
    )
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
