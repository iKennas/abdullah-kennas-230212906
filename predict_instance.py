"""Run Mask R-CNN on a single image and save an annotated overlay.

Usage:
    python predict_instance.py                       # random patch from test
    python predict_instance.py --image path/to.jpg   # custom image
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from build_model_instance import get_maskrcnn


_IMG_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def pick_random_test_image() -> Path:
    """Pick a random patch from ``data/test/images/``."""
    test_dir = config.TEST_IMAGES_DIR
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {test_dir}")
    candidates = [
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No images in {test_dir}")
    return random.Random().choice(candidates)


def load_model(device: str) -> torch.nn.Module:
    ckpt_path = config.INSTANCE_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Mask R-CNN checkpoint not found: {ckpt_path}\n"
            "Train it first with: python train_instance.py"
        )
    model = get_maskrcnn(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image_rgb: np.ndarray, device: str,
            score_thr: float) -> dict:
    image_t = torch.from_numpy(
        image_rgb.transpose(2, 0, 1).copy()
    ).float().to(device) / 255.0
    out = model([image_t])[0]
    boxes  = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    masks  = (out["masks"].cpu().numpy() > 0.5).astype(np.uint8)[:, 0]
    keep = scores >= score_thr
    return {
        "boxes":  boxes[keep],
        "scores": scores[keep],
        "masks":  masks[keep],
    }


def render_overlay(image_rgb: np.ndarray, pred: dict) -> np.ndarray:
    """Blend per-instance coloured masks + box outlines on the RGB image."""
    out = image_rgb.astype(np.float32)
    overlay = np.zeros_like(out)
    rng = np.random.default_rng(42)
    for i in range(pred["masks"].shape[0]):
        color = rng.random(3) * 200 + 55  # avoid very dark colours
        for c in range(3):
            overlay[..., c] += pred["masks"][i] * color[c]
    out = np.clip(out * 0.55 + overlay * 0.45, 0, 255).astype(np.uint8)

    for i in range(pred["boxes"].shape[0]):
        x1, y1, x2, y2 = pred["boxes"][i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = f"{pred['scores'][i]:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 3, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def save_figure(image_rgb: np.ndarray, overlay: np.ndarray, pred: dict,
                out_path: Path, score_thr: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Mask R-CNN - {pred['boxes'].shape[0]} buildings (score >= {score_thr})"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image Mask R-CNN inference."
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--score-thr", type=float,
                        default=config.INSTANCE_SCORE_THRESHOLD)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    config.ensure_dirs()

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    else:
        image_path = pick_random_test_image()
        print(f"No --image: using a random test patch: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"  image size: {image_rgb.shape[1]} x {image_rgb.shape[0]}")

    model = load_model(config.DEVICE)
    pred = predict(model, image_rgb, config.DEVICE, args.score_thr)
    print(f"  buildings detected (score >= {args.score_thr}): "
          f"{pred['boxes'].shape[0]}")

    overlay = render_overlay(image_rgb, pred)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = config.PREDICTIONS_DIR / f"{image_path.stem}_maskrcnn_pred.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(image_rgb, overlay, pred, out_path, args.score_thr)


if __name__ == "__main__":
    main()
