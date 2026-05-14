"""Test-set metrics and save example prediction figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from build_model import get_model
from dataset import (
    BuildingDataset,
    get_preprocessing,
    get_validation_augmentation,
)
from utils.metrics import (
    compute_all_metrics,
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str) -> torch.nn.Module:
    """Build the architecture and load the best checkpoint from disk.

    Args:
        model_name: one of ``config.SUPPORTED_MODELS``.
        device: target device (``"cuda"`` or ``"cpu"``).

    Returns:
        Model in ``eval`` mode, ready for inference.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
    """
    ckpt_path = config.checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}: {ckpt_path}\n"
            "Train the model first with: python train.py --model "
            f"{model_name}"
        )
    model = get_model(model_name).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_test_dataset() -> Tuple[BuildingDataset, BuildingDataset]:
    """Build two views of the test set:

    * ``ds_eval``  - normalised tensors used for inference.
    * ``ds_visu``  - raw uint8 RGB used to render visualisations
      (we cannot un-normalise float tensors safely).

    Returns:
        ``(ds_eval, ds_visu)`` - both indexed by the same integer.
    """
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config.ENCODER_NAME, pretrained=config.ENCODER_WEIGHTS,
    )
    ds_eval = BuildingDataset(
        config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    ds_visu = BuildingDataset(
        config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR,
        augmentation=get_validation_augmentation(),
        preprocessing=None,
    )
    return ds_eval, ds_visu


@torch.no_grad()
def run_inference(model: torch.nn.Module, dataset: BuildingDataset,
                  device: str, batch_size: int = config.BATCH_SIZE
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Predict every sample of ``dataset`` and aggregate metrics.

    Args:
        model: trained network.
        dataset: dataset returning normalised tensors.
        device: target device.
        batch_size: mini-batch size used for inference.

    Returns:
        ``(per_image_iou, all_logits_cpu, all_targets_cpu, metric_dict)``
        - ``per_image_iou``: shape ``(N,)`` IoU score per image.
        - ``all_logits_cpu``: shape ``(N, 1, H, W)`` raw logits.
        - ``all_targets_cpu``: shape ``(N, 1, H, W)`` ground truth masks.
        - ``metric_dict``: aggregated metrics (mean over the test set).
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    per_image_iou: List[float] = []

    for images, masks in tqdm(loader, desc="Test inference", unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)

        # Per-image IoU (loop over the batch -> small overhead, clearer code).
        for i in range(images.size(0)):
            per_image_iou.append(
                iou_score(logits[i:i + 1], masks[i:i + 1])
            )

        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(logits_cat, targets_cat)
    return np.array(per_image_iou), logits_cat, targets_cat, metrics


def print_metrics_table(model_name: str, metrics: Dict[str, float]) -> None:
    """Pretty-print the aggregated metrics table.

    Args:
        model_name: architecture name to show in the header.
        metrics: dictionary returned by :func:`compute_all_metrics`.
    """
    print()
    print("=" * 60)
    print(f"Test results - {model_name}")
    print("=" * 60)
    print(f"{'Metric':<20}{'Value':>12}")
    print("-" * 60)
    print(f"{'IoU (Jaccard)':<20}{metrics['iou']:>12.4f}")
    print(f"{'Dice (F1)':<20}{metrics['dice']:>12.4f}")
    print(f"{'Pixel accuracy':<20}{metrics['accuracy']:>12.4f}")
    print(f"{'Precision':<20}{metrics['precision']:>12.4f}")
    print(f"{'Recall':<20}{metrics['recall']:>12.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _logit_to_mask(logit: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid + threshold a single logit map to a uint8 ``{0,255}`` mask."""
    if logit.dim() == 3:
        logit = logit[0]
    prob = torch.sigmoid(logit).cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    return mask


def _overlay(image_rgb: np.ndarray, mask: np.ndarray,
             color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4
             ) -> np.ndarray:
    """Blend a binary mask onto an RGB image using a colour overlay.

    Args:
        image_rgb: HxWx3 uint8 image.
        mask: HxW uint8 mask in ``{0, 255}``.
        color: overlay RGB colour (default red).
        alpha: opacity of the overlay (0=transparent, 1=opaque).

    Returns:
        HxWx3 uint8 image with the overlay.
    """
    overlay = image_rgb.copy()
    coloured = np.zeros_like(image_rgb)
    coloured[..., 0] = color[0]
    coloured[..., 1] = color[1]
    coloured[..., 2] = color[2]

    mask_bool = mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * image_rgb[mask_bool] + alpha * coloured[mask_bool]
    ).astype(np.uint8)
    return overlay


def plot_grid(indices: List[int], ds_visu: BuildingDataset,
              logits: torch.Tensor, targets: torch.Tensor,
              out_path: Path, title: str,
              ious: np.ndarray | None = None) -> None:
    """Render a 4-column grid: image | GT | prediction | overlay.

    Args:
        indices: list of dataset indices to plot (one row per sample).
        ds_visu: dataset returning *raw* RGB tensors for visualisation.
        logits: tensor of shape ``(N, 1, H, W)``.
        targets: tensor of shape ``(N, 1, H, W)``.
        out_path: PNG destination.
        title: figure suptitle.
        ious: optional per-image IoU array, used to label rows.
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        # ds_visu returns a CHW float tensor in [0, 1]
        image_t, _ = ds_visu[idx]
        image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        gt = (targets[idx, 0].numpy() > 0.5).astype(np.uint8) * 255
        pred = _logit_to_mask(logits[idx])
        ov = _overlay(image, pred)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Image")
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(ov)
        title_overlay = "Overlay"
        if ious is not None:
            title_overlay += f"  (IoU={ious[idx]:.3f})"
        axes[row, 3].set_title(title_overlay)
        for c in range(4):
            axes[row, c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_iou_histogram(ious: np.ndarray, out_path: Path, model_name: str) -> None:
    """Plot histogram of per-image IoUs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=30, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(ious.mean(), color="red", linestyle="--",
               label=f"mean = {ious.mean():.3f}")
    ax.axvline(np.median(ious), color="green", linestyle="--",
               label=f"median = {np.median(ious):.3f}")
    ax.set_xlabel("Per-image IoU")
    ax.set_ylabel("Number of test patches")
    ax.set_title(f"Distribution of per-image IoU on the test set ({model_name})")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--model", type=str, default="unet",
        choices=list(config.SUPPORTED_MODELS),
        help="Architecture to evaluate.",
    )
    parser.add_argument(
        "--num-grid", type=int, default=10,
        help="Number of random samples in the grid figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    config.ensure_dirs()

    device = config.DEVICE
    model = load_model(args.model, device)
    ds_eval, ds_visu = build_test_dataset()
    print(f"Test set size: {len(ds_eval)} patches\n")

    per_image_iou, logits, targets, metrics = run_inference(model, ds_eval, device)
    print_metrics_table(args.model, metrics)

    # ---- Visualisations -------------------------------------------------
    safe = args.model.replace("+", "plus")
    rng = np.random.default_rng(config.SEED)
    grid_indices = rng.choice(
        len(ds_eval), size=min(args.num_grid, len(ds_eval)), replace=False,
    ).tolist()

    plot_grid(
        grid_indices, ds_visu, logits, targets,
        out_path=config.PREDICTIONS_DIR / f"{safe}_grid.png",
        title=f"Random test samples - {args.model}",
        ious=per_image_iou,
    )

    sorted_idx = np.argsort(per_image_iou)
    worst_idx = sorted_idx[:5].tolist()
    best_idx = sorted_idx[-5:][::-1].tolist()

    plot_grid(
        best_idx, ds_visu, logits, targets,
        out_path=config.PREDICTIONS_DIR / f"{safe}_top5_best.png",
        title=f"Top-5 BEST predictions - {args.model}",
        ious=per_image_iou,
    )
    plot_grid(
        worst_idx, ds_visu, logits, targets,
        out_path=config.PREDICTIONS_DIR / f"{safe}_top5_worst.png",
        title=f"Top-5 WORST predictions - {args.model}",
        ious=per_image_iou,
    )

    plot_iou_histogram(
        per_image_iou,
        config.PREDICTIONS_DIR / f"{safe}_iou_histogram.png",
        args.model,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
