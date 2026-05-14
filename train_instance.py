"""Train Mask R-CNN on the Massachusetts Buildings patches.

Usage:
    python train_instance.py
    python train_instance.py --epochs 20 --batch-size 4 --lr 5e-4

Torchvision detection models return a dict of losses during ``model.train()``;
we sum them and back-propagate. Validation uses the same model in ``eval``
mode but with ``targets`` supplied so we can keep a held-out loss estimate;
COCO mAP is computed by ``evaluate_instance.py`` after training.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from build_model_instance import count_parameters, get_maskrcnn
from dataset_instance import get_instance_loaders


@dataclass
class InstanceTrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss:   List[float] = field(default_factory=list)
    lr:         List[float] = field(default_factory=list)


def _move_targets(targets, device: str):
    """Push every tensor in each target dict to ``device``."""
    return [
        {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
         for k, v in t.items()}
        for t in targets
    ]


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str, epoch: int,
                    grad_clip: float = 5.0) -> float:
    """Single training epoch; returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = _move_targets(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not math.isfinite(loss.item()):
            pbar.write(f"  [WARN] non-finite loss {loss.item()}, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: str,
             epoch: int) -> float:
    """Validation loss: we keep the model in ``train`` mode so torchvision
    still returns the loss dict, but we disable gradient computation."""
    model.train()  # required for loss_dict; outputs are not used elsewhere
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = _move_targets(targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        total_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")

    return total_loss / max(n_batches, 1)


def plot_history(history: InstanceTrainHistory, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_title("Mask R-CNN - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss (sum of 5 heads)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.lr, color="tab:purple")
    axes[1].set_title("Learning rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Mask R-CNN training curves", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "maskrcnn_training_curves.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  curves saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on Massachusetts Buildings."
    )
    parser.add_argument("--epochs", type=int,
                        default=config.INSTANCE_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int,
                        default=config.INSTANCE_BATCH_SIZE)
    parser.add_argument("--lr", type=float,
                        default=config.INSTANCE_LEARNING_RATE)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not load the COCO pretrained weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    config.ensure_dirs()

    print("=" * 70)
    print("Training Mask R-CNN (instance segmentation)")
    print("=" * 70)
    print(f"  device         : {config.DEVICE}")
    print(f"  epochs (max)   : {args.epochs}")
    print(f"  batch size     : {args.batch_size}")
    print(f"  learning rate  : {args.lr}")
    print(f"  pretrained     : {not args.no_pretrained}")
    print()

    # ---- Data -----------------------------------------------------------
    train_loader, val_loader, _ = get_instance_loaders(
        batch_size=args.batch_size,
    )
    print(f"  train batches  : {len(train_loader)}")
    print(f"  val   batches  : {len(val_loader)}")

    # ---- Model ----------------------------------------------------------
    model = get_maskrcnn(pretrained=not args.no_pretrained).to(config.DEVICE)
    n_params = count_parameters(model)
    print(f"  trainable params: {n_params:,} ({n_params / 1e6:.2f} M)\n")

    # ---- Optim / Scheduler ---------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=max(args.epochs // 3, 1),
                       gamma=0.5)

    # ---- Logging --------------------------------------------------------
    writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR / "maskrcnn"))
    history = InstanceTrainHistory()
    best_val = math.inf
    ckpt_path: Path = config.INSTANCE_CHECKPOINT

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, config.DEVICE, epoch,
            )
            val_loss = validate(model, val_loader, config.DEVICE, epoch)
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.lr.append(current_lr)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val",   val_loss,   epoch)
            writer.add_scalar("hp/lr",      current_lr, epoch)

            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={current_lr:.2e} | "
                f"{elapsed:5.1f}s"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_name": "maskrcnn_resnet50_fpn_v2",
                    "num_classes": config.INSTANCE_NUM_CLASSES,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                }, ckpt_path)
                print(f"   -> new best (val_loss={best_val:.4f}) saved to "
                      f"{ckpt_path.name}")

    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA out of memory. Try --batch-size 2.")
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        writer.close()

    print("\nSaving training curves...")
    plot_history(history, config.PLOTS_DIR)
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint   : {ckpt_path}")
    print("\nNext: python evaluate_instance.py")


if __name__ == "__main__":
    main()
