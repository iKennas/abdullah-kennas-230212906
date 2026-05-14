"""Tile Kaggle Mass. Buildings (train/val/test) into 256px patches under data/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import config


# Image extensions accepted on disk for both images and masks.
_IMG_EXTENSIONS: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_images(directory: Path) -> List[Path]:
    """Sorted list of raster files in ``directory``."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )


def _index_by_stem(paths: List[Path]) -> Dict[str, Path]:
    """stem -> path (last duplicate wins)."""
    return {p.stem: p for p in paths}


def _find_split_root(raw_dir: Path, split: str) -> Optional[Tuple[Path, Path]]:
    """Guess ``images`` / ``masks`` dirs for split (handles common Kaggle layouts)."""
    parents = [
        raw_dir,
        raw_dir / "png",
        raw_dir / "tiff",
        raw_dir / "Massachusetts",
    ]

    def _has_images(p: Path) -> bool:
        return p.is_dir() and any(_list_images(p))

    for parent in parents:
        # Layout A — "{split}/images" + "{split}/masks"
        a_img = parent / split / "images"
        a_msk = parent / split / "masks"
        if _has_images(a_img) and a_msk.is_dir():
            return a_img, a_msk

        # Layout B — "{split}" + "{split}_labels"
        b_img = parent / split
        b_msk = parent / f"{split}_labels"
        if _has_images(b_img) and b_msk.is_dir():
            return b_img, b_msk

    # Final fallback: recursive search.
    for candidate in raw_dir.rglob(f"{split}_labels"):
        if candidate.is_dir() and _has_images(candidate.parent / split):
            return candidate.parent / split, candidate
    for candidate in raw_dir.rglob(f"{split}/images"):
        msk = candidate.parent / "masks"
        if msk.is_dir() and _has_images(candidate):
            return candidate, msk
    return None


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Turn mask into uint8 with only 0 and 255."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return ((mask > 0).astype(np.uint8)) * 255


def _tile_pair(image: np.ndarray, mask: np.ndarray, patch_size: int
               ) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Non-overlapping ``patch_size`` crops; drops remainders."""
    h, w = image.shape[:2]
    n_rows = h // patch_size
    n_cols = w // patch_size

    out: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * patch_size, (r + 1) * patch_size
            x0, x1 = c * patch_size, (c + 1) * patch_size
            out.append((r, c, image[y0:y1, x0:x1], mask[y0:y1, x0:x1]))
    return out


def _clean_dir(directory: Path) -> None:
    """mkdir + wipe files inside (fresh patch export)."""
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.iterdir():
        if f.is_file():
            f.unlink()


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def _process_split(split: str, raw_root: Path, out_img_dir: Path,
                   out_mask_dir: Path, patch_size: int
                   ) -> Dict[str, int]:
    """Dump patches for ``split`` and return simple counters."""
    stats = {
        "source_images": 0,
        "patches": 0,
        "patches_with_buildings": 0,
        "total_pixels": 0,
        "foreground_pixels": 0,
    }

    located = _find_split_root(raw_root, split)
    if located is None:
        print(f"  [WARN] No '{split}' split found under {raw_root}. Skipping.")
        return stats
    src_img_dir, src_mask_dir = located

    images = _list_images(src_img_dir)
    masks_by_stem = _index_by_stem(_list_images(src_mask_dir))
    if not images:
        print(f"  [WARN] No images found in {src_img_dir}. Skipping '{split}'.")
        return stats

    _clean_dir(out_img_dir)
    _clean_dir(out_mask_dir)

    pbar = tqdm(images, desc=f"Tiling {split}", unit="img")
    for img_path in pbar:
        # Match image -> mask by file STEM so different extensions are fine
        # (e.g. images stored as .tiff, masks as .tif).
        mask_path = masks_by_stem.get(img_path.stem)
        if mask_path is None:
            pbar.write(f"  [WARN] No mask for {img_path.name}, skipping.")
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            pbar.write(f"  [WARN] Failed to read {img_path.name}, skipping.")
            continue

        mask = _binarize_mask(mask)

        # If shapes diverge for any reason (rare), align mask to image with
        # nearest-neighbour interpolation to preserve binary values.
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        stats["source_images"] += 1
        for r, c, img_patch, mask_patch in _tile_pair(image, mask, patch_size):
            patch_name = f"{img_path.stem}_patch_{r:02d}_{c:02d}.png"
            cv2.imwrite(str(out_img_dir / patch_name), img_patch)
            cv2.imwrite(str(out_mask_dir / patch_name), mask_patch)

            stats["patches"] += 1
            n_fg = int((mask_patch > 0).sum())
            stats["foreground_pixels"] += n_fg
            stats["total_pixels"] += mask_patch.size
            if n_fg > 0:
                stats["patches_with_buildings"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile the Kaggle Massachusetts Buildings Dataset into "
            "256x256 patches for training."
        )
    )
    parser.add_argument(
        "--raw_dir", type=str, default=None,
        help=(
            "Path to the extracted Kaggle dataset folder (the one that "
            "contains 'train/', 'val/', 'test/' subfolders). If omitted, "
            "uses config.RAW_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--patch_size", type=int, default=config.IMAGE_SIZE,
        help=f"Square patch edge length (default: {config.IMAGE_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    config.ensure_dirs()

    raw_candidate = args.raw_dir or config.RAW_DATA_DIR
    raw_root = (
        Path(raw_candidate).expanduser().resolve()
        if raw_candidate is not None and str(raw_candidate).strip()
        else None
    )
    patch_size = int(args.patch_size)

    print("=" * 70)
    print("Aerial Building Segmentation - Data Preparation")
    print("=" * 70)
    print(f"Raw dataset : {raw_root if raw_root else '(not provided)'}")
    print(f"Patch size  : {patch_size}x{patch_size}")
    print(f"Output root : {config.DATA_DIR}")
    print()

    if raw_root is None or not raw_root.exists():
        print(
            "[ERROR] --raw_dir not provided or does not exist.\n"
            "  Pass it explicitly:\n"
            "    python prepare_data.py --raw_dir \"PATH_TO_EXTRACTED_FOLDER\"\n"
            "  ...or set RAW_DATA_DIR in config.py."
        )
        sys.exit(1)

    splits = [
        ("train", config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR),
        ("val",   config.VAL_IMAGES_DIR,   config.VAL_MASKS_DIR),
        ("test",  config.TEST_IMAGES_DIR,  config.TEST_MASKS_DIR),
    ]

    summary = []
    for name, out_img, out_mask in splits:
        stats = _process_split(name, raw_root, out_img, out_mask, patch_size)
        summary.append((name, stats))

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(
        f"{'Split':<8}{'Src imgs':>10}{'Patches':>10}"
        f"{'% w/ bldg':>13}{'% bldg px':>13}{'% bg px':>13}"
    )
    print("-" * 78)
    total_patches = 0
    for name, s in summary:
        n_patch = s["patches"]
        total_patches += n_patch
        pct_with = (s["patches_with_buildings"] / n_patch * 100.0) if n_patch else 0.0
        pct_fg = (s["foreground_pixels"] / s["total_pixels"] * 100.0) \
            if s["total_pixels"] else 0.0
        pct_bg = 100.0 - pct_fg
        print(
            f"{name:<8}{s['source_images']:>10}{n_patch:>10}"
            f"{pct_with:>12.2f}%{pct_fg:>12.2f}%{pct_bg:>12.2f}%"
        )
    print("-" * 78)
    print(f"{'TOTAL':<8}{'':>10}{total_patches:>10}")
    print("=" * 78)

    if total_patches == 0:
        print(
            "\n[ERROR] No patches were produced. Verify that --raw_dir points "
            "to the extracted dataset and that train/val/test subfolders exist."
        )
        sys.exit(1)

    print("\nDone. You can now run: python train.py --model unet")


if __name__ == "__main__":
    main()
