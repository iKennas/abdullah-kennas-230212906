"""Train U-Net / U-Net++ / DeepLabV3+ on Mass. Buildings patches.

Examples:
    python train.py --model unet
    python train.py --model unet++
    python train.py --model deeplabv3+

Loss is Dice + BCE on logits, Adam + ReduceLROnPlateau, early stopping on val
loss, best checkpoint by val IoU. Logs to TensorBoard and saves curve plots.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from build_model import count_parameters, get_model
from dataset import get_loaders
from utils.metrics import dice_coefficient, iou_score


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """BCE-with-logits + SMP binary DiceLoss (weights default 0.5 / 0.5)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + \
               self.dice_weight * self.dice(logits, target)


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------

@dataclass
class TrainHistory:
    """Per-epoch scalars."""
    train_loss: List[float] = field(default_factory=list)
    val_loss:   List[float] = field(default_factory=list)
    val_iou:    List[float] = field(default_factory=list)
    val_dice:   List[float] = field(default_factory=list)
    lr:         List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: str, epoch: int) -> float:
    """One training epoch; returns mean loss."""
    model.train()
    running_loss = 0.0
    running_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_count += bs
        pbar.set_postfix(loss=f"{running_loss / running_count:.4f}")

    return running_loss / max(running_count, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str, epoch: int) -> Dict[str, float]:
    """Validation pass; dict with loss, iou, dice."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou += iou_score(logits, masks) * bs
        total_dice += dice_coefficient(logits, masks) * bs
        n += bs

        pbar.set_postfix(
            loss=f"{total_loss / n:.4f}",
            iou=f"{total_iou / n:.4f}",
        )

    return {
        "loss": total_loss / max(n, 1),
        "iou":  total_iou / max(n, 1),
        "dice": total_dice / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history: TrainHistory, model_name: str, out_dir: Path) -> None:
    """Save loss / IoU / Dice plots to outputs/plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.val_iou, label="val IoU", color="tab:green")
    axes[1].set_title(f"{model_name} - Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history.val_dice, label="val Dice", color="tab:orange")
    axes[2].set_title(f"{model_name} - Validation Dice")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Training curves ({model_name})", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / f"{model_name.replace('+', 'plus')}_training_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  curves saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a segmentation model on Massachusetts Buildings."
    )
    parser.add_argument(
        "--model", type=str, default="unet",
        choices=list(config.SUPPORTED_MODELS),
        help="Architecture to train.",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help=f"Maximum number of epochs (default {config.NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
        help=f"Batch size (default {config.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help=f"Initial learning rate (default {config.LEARNING_RATE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    config.ensure_dirs()

    print("=" * 70)
    print(f"Training: {args.model.upper()}")
    print("=" * 70)
    print(f"  device      : {config.DEVICE}")
    print(f"  epochs (max): {args.epochs}")
    print(f"  batch size  : {args.batch_size}")
    print(f"  learning rate: {args.lr}")
    print()

    # ---- Data ------------------------------------------------------------
    train_loader, val_loader, _ = get_loaders(batch_size=args.batch_size)
    print(f"  train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # ---- Model -----------------------------------------------------------
    model = get_model(args.model).to(config.DEVICE)
    n_params = count_parameters(model)
    print(f"  trainable params: {n_params:,} ({n_params / 1e6:.2f} M)")

    # ---- Loss / Optim / Scheduler ---------------------------------------
    criterion = CombinedLoss().to(config.DEVICE)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    # ---- Logging ---------------------------------------------------------
    safe_name = args.model.replace("+", "plus")
    tb_dir = config.TENSORBOARD_DIR / safe_name
    writer = SummaryWriter(log_dir=str(tb_dir))

    history = TrainHistory()
    best_iou = -1.0
    epochs_no_improve = 0
    ckpt_path = config.checkpoint_path(args.model)

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE, epoch,
            )
            val_metrics = validate(
                model, val_loader, criterion, config.DEVICE, epoch,
            )
            elapsed = time.time() - t0

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["loss"])

            history.train_loss.append(train_loss)
            history.val_loss.append(val_metrics["loss"])
            history.val_iou.append(val_metrics["iou"])
            history.val_dice.append(val_metrics["dice"])
            history.lr.append(current_lr)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val",   val_metrics["loss"], epoch)
            writer.add_scalar("metric/iou", val_metrics["iou"], epoch)
            writer.add_scalar("metric/dice", val_metrics["dice"], epoch)
            writer.add_scalar("hp/lr", current_lr, epoch)

            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_dice={val_metrics['dice']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"{elapsed:5.1f}s"
            )

            # ---- Checkpoint best model by val IoU --------------------------
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                epochs_no_improve = 0
                torch.save({
                    "model_name": args.model,
                    "encoder_name": config.ENCODER_NAME,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                    "val_dice": val_metrics["dice"],
                }, ckpt_path)
                print(f"   -> new best (val_iou={best_iou:.4f}) saved to {ckpt_path.name}")
            else:
                epochs_no_improve += 1

            # ---- Early stopping --------------------------------------------
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping triggered: val_iou did not improve "
                    f"for {config.EARLY_STOP_PATIENCE} epochs."
                )
                break

    except torch.cuda.OutOfMemoryError:
        print(
            "\n[ERROR] CUDA out of memory. Try reducing --batch-size "
            "or IMAGE_SIZE in config.py."
        )
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        writer.close()

    # ---- Final plots -----------------------------------------------------
    print("\nSaving training curves...")
    plot_history(history, args.model, config.PLOTS_DIR)
    print(f"\nBest val IoU: {best_iou:.4f}")
    print(f"Checkpoint  : {ckpt_path}")


if __name__ == "__main__":
    main()
