"""Load a checkpoint, run one image (optional tile+stitch), save overlay."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

import config
from build_model import get_model


_TEST_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png", ".tif", ".tiff", ".jpg", ".jpeg",
)


def pick_random_test_image() -> Path:
    """Choose a random image file from ``data/test/images/``.

    Used when ``predict.py`` is run without ``--image`` so you can smoke-test
    a trained checkpoint immediately after ``prepare_data.py``.

    Returns:
        Absolute path to a test-set patch image.

    Raises:
        FileNotFoundError: if the folder is missing or empty.
    """
    test_dir = config.TEST_IMAGES_DIR
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test image directory does not exist: {test_dir}\n"
            "Run prepare_data.py first to populate data/test/images/."
        )
    candidates = [
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _TEST_IMAGE_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No image files ({', '.join(_TEST_IMAGE_EXTENSIONS)}) found in "
            f"{test_dir}. Run prepare_data.py first."
        )
    # Fresh RNG so each run picks a different patch while training stays seeded.
    return random.Random().choice(candidates)


def load_model(model_name: str, device: str) -> torch.nn.Module:
    """Load a trained model from its best checkpoint.

    Args:
        model_name: one of the supported architectures.
        device: target device.

    Returns:
        Model in eval mode.
    """
    ckpt_path = config.checkpoint_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first with: python train.py --model {model_name}"
        )
    model = get_model(model_name).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _normalize_for_encoder(image_rgb_uint8: np.ndarray) -> np.ndarray:
    """Apply SMP's ImageNet normalisation for the project's encoder."""
    fn = smp.encoders.get_preprocessing_fn(
        config.ENCODER_NAME, pretrained=config.ENCODER_WEIGHTS,
    )
    return fn(image_rgb_uint8)


@torch.no_grad()
def predict_image(model: torch.nn.Module, image_rgb: np.ndarray,
                  device: str, patch_size: int = config.IMAGE_SIZE,
                  stride: int | None = None,
                  threshold: float = 0.5
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a (possibly large) RGB image.

    Args:
        model: trained network in eval mode.
        image_rgb: HxWx3 uint8 array.
        device: target device.
        patch_size: tile edge length used for inference.
        stride: stride between consecutive tiles; defaults to ``patch_size//2``.
        threshold: probability threshold to binarise the output.

    Returns:
        ``(prob_map, binary_mask)``
        - ``prob_map``: HxW float32 probabilities in [0, 1].
        - ``binary_mask``: HxW uint8 mask in {0, 255}.
    """
    if stride is None:
        stride = patch_size // 2

    # Remember the user's original size - we crop back to it at the end.
    orig_h, orig_w = image_rgb.shape[:2]

    h, w = image_rgb.shape[:2]
    # If the image is smaller than a patch, pad up to ``patch_size``.
    pad_bottom = max(0, patch_size - h)
    pad_right = max(0, patch_size - w)
    if pad_bottom or pad_right:
        image_rgb = cv2.copyMakeBorder(
            image_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101,
        )
        h, w = image_rgb.shape[:2]

    # Pad on the right/bottom so that (size - patch_size) is a multiple of stride.
    pad_h = (stride - (h - patch_size) % stride) % stride if h > patch_size else 0
    pad_w = (stride - (w - patch_size) % stride) % stride if w > patch_size else 0
    if pad_h or pad_w:
        image_rgb = cv2.copyMakeBorder(
            image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101,
        )
    H, W = image_rgb.shape[:2]

    prob_accum = np.zeros((H, W), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)

    # Iterate over all patch top-left corners.
    ys = list(range(0, H - patch_size + 1, stride)) if H > patch_size else [0]
    xs = list(range(0, W - patch_size + 1, stride)) if W > patch_size else [0]
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    print(f"  Inference on {len(ys) * len(xs)} patch(es) "
          f"({patch_size}x{patch_size}, stride={stride})...")

    for y in ys:
        for x in xs:
            patch = image_rgb[y:y + patch_size, x:x + patch_size]
            patch_norm = _normalize_for_encoder(patch)
            tensor = torch.from_numpy(
                patch_norm.transpose(2, 0, 1)
            ).unsqueeze(0).float().to(device)

            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            prob_accum[y:y + patch_size, x:x + patch_size] += prob
            weight_accum[y:y + patch_size, x:x + patch_size] += 1.0

    # Average the overlapping predictions.
    prob_map = prob_accum / np.maximum(weight_accum, 1e-6)

    # Crop back to the user's original input size (undo BOTH paddings).
    prob_map = prob_map[:orig_h, :orig_w]
    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    return prob_map, binary_mask


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = (255, 0, 0),
                 alpha: float = 0.4) -> np.ndarray:
    """Blend a mask on an image with a coloured semi-transparent overlay."""
    overlay = image_rgb.copy()
    coloured = np.zeros_like(image_rgb)
    coloured[..., 0], coloured[..., 1], coloured[..., 2] = color
    sel = mask > 0
    overlay[sel] = (
        (1 - alpha) * image_rgb[sel] + alpha * coloured[sel]
    ).astype(np.uint8)
    return overlay


def count_buildings(binary_mask: np.ndarray, min_area: int = 30) -> int:
    """Count connected components in a binary mask.

    Args:
        binary_mask: HxW uint8 mask in {0, 255}.
        min_area: ignore components smaller than this many pixels (noise).

    Returns:
        Number of detected buildings.
    """
    n_labels, labels = cv2.connectedComponents((binary_mask > 0).astype(np.uint8))
    count = 0
    for lbl in range(1, n_labels):  # 0 is background
        area = int((labels == lbl).sum())
        if area >= min_area:
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict building masks for a single image.")
    parser.add_argument(
        "--image", type=str, default=None,
        help=(
            "Path to the input image. If omitted, a random patch from "
            "data/test/images/ is used."
        ),
    )
    parser.add_argument("--model", type=str, default="unet",
                        choices=list(config.SUPPORTED_MODELS),
                        help="Architecture to use for prediction.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binarisation.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride between patches (default IMAGE_SIZE/2).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: outputs/predictions/).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    config.ensure_dirs()

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        print(f"Input image (--image): {image_path}")
    else:
        image_path = pick_random_test_image()
        print(
            "No --image argument: using a random patch from the test set:\n"
            f"  {image_path.resolve()}"
        )

    print(f"Loading model {args.model}...")
    model = load_model(args.model, config.DEVICE)

    print(f"Reading image {image_path} ...")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"  size: {image_rgb.shape[1]} x {image_rgb.shape[0]} (W x H)")

    prob_map, binary_mask = predict_image(
        model, image_rgb, config.DEVICE,
        patch_size=config.IMAGE_SIZE,
        stride=args.stride,
        threshold=args.threshold,
    )

    n_buildings = count_buildings(binary_mask)
    print(f"  Detected ~{n_buildings} buildings (connected components).")

    overlay = overlay_mask(image_rgb, binary_mask)

    safe = args.model.replace("+", "plus")
    if args.output is None:
        out_path = config.PREDICTIONS_DIR / f"{image_path.stem}_{safe}_pred.png"
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save overlay (convert RGB -> BGR for cv2.imwrite).
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    mask_path = out_path.with_name(out_path.stem + "_mask.png")
    cv2.imwrite(str(mask_path), binary_mask)

    print(f"  Overlay saved to: {out_path}")
    print(f"  Mask    saved to: {mask_path}")


if __name__ == "__main__":
    main()
