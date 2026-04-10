import argparse
import os
from pathlib import Path

import mlflow
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from src.datasets.mp16 import GeoTIRDataset, PairAwareBatchSampler, geo_collate_fn
from src.geotir.model import GeoTIRModel

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
MLFLOW_TRACKING_URI = "http://localhost:5555"
EXPERIMENT_NAME = "geotir-clip-finetune"


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    train_df = pl.read_csv(args.data)

    dataset = GeoTIRDataset(
        df=train_df,
        base_img_path=args.img_path,
        processor=processor,
    )
    sampler = PairAwareBatchSampler(
        df=train_df,
        batch_size=args.batch_size,
        drop_last=True,
        cap=args.pair_cap,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=geo_collate_fn)

    model = GeoTIRModel(
        clip_model_name=CLIP_MODEL_NAME,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = len(sampler) * args.epochs
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
        "max_text_length": 32,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "scheduler": "cosine",
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    ckpt_dir = Path("checkpoints") / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(hparams)
        print(f"MLflow run: {run.info.run_id}")
        print(f"Checkpoints: {ckpt_dir}")

        for epoch in tqdm(range(1, args.epochs + 1), desc="finetune"):
            model.train()
            trloss = 0.0
            pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)

            for batch in pbar:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                output = model(batch)
                loss = output["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                trloss += loss.item()
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                mlflow.log_metrics(
                    {
                        "loss": loss.item(),
                        "temperature": output["temperature"],
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

            epoch_loss = trloss / len(loader)
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            tqdm.write(f"epoch {epoch} | loss {epoch_loss:.4f}")

            # Save per-epoch checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:02d}.pt")

            # Save best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(ckpt, ckpt_dir / "best.pt")
                tqdm.write(f"  -> best saved (loss {best_loss:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--run-name", default="clip-vit-large-lora-r16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--pair-cap", type=int, default=1157, help="Max pairs per group per epoch")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of total steps used for linear warmup")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
