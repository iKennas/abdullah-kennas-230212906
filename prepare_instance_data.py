"""Convert the existing binary masks into COCO-style instance annotations.

For each split (train / val / test) this script walks
``data/{split}/masks/``, runs connected-component labelling on every patch
mask, and writes one COCO JSON per split at ``data/{split}/instances.json``:

* Each connected component (>= ``--min-area`` pixels) becomes one annotation
  for the class ``building`` (category id 1; 0 is reserved for background by
  COCO convention).
* The segmentation is encoded as RLE via ``pycocotools.mask.encode`` for
  compactness; ``bbox`` uses the COCO ``[x, y, w, h]`` format.

Touching buildings inside the *same* mask blob remain a single instance
(known limitation of binary semantic masks). Use ``--separate-touching`` to
erode the mask before labelling, which splits some neighbours apart at the
cost of dropping tiny components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

import config


_IMG_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list_files(directory: Path) -> List[Path]:
    """Sorted list of raster files in a directory."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Force the mask to a 0/1 uint8 single-channel image."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def _maybe_separate_touching(mask: np.ndarray) -> np.ndarray:
    """Mild morphological erosion to break thin connections between blobs."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(mask, kernel, iterations=1)


def _instances_from_mask(binary_mask: np.ndarray, min_area: int
                         ) -> List[Tuple[List[int], np.ndarray]]:
    """Return ``[(bbox_xywh, instance_mask_uint8), ...]`` from a binary mask.

    Uses ``cv2.connectedComponentsWithStats`` so the bounding boxes are
    computed for free.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8,
    )
    instances: List[Tuple[List[int], np.ndarray]] = []
    # Label 0 is background.
    for lbl in range(1, n_labels):
        x, y, w, h, area = stats[lbl]
        if area < min_area or w < 1 or h < 1:
            continue
        inst_mask = (labels == lbl).astype(np.uint8)
        instances.append(([int(x), int(y), int(w), int(h)], inst_mask))
    return instances


def _encode_rle(instance_mask: np.ndarray) -> Dict:
    """RLE-encode a HxW uint8 mask in a COCO-JSON-friendly dict."""
    rle = mask_utils.encode(np.asfortranarray(instance_mask))
    # ``counts`` comes back as bytes; JSON only takes str.
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _process_split(split_name: str, images_dir: Path, masks_dir: Path,
                   out_json: Path, min_area: int,
                   separate_touching: bool) -> Dict[str, int]:
    """Build the COCO JSON for one split and return summary counters."""
    images = _list_files(images_dir)
    masks_by_stem = {p.stem: p for p in _list_files(masks_dir)}

    coco: Dict = {
        "info": {
            "description": f"Massachusetts Buildings - instance annotations ({split_name})",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "building", "supercategory": "structure"},
        ],
        "images": [],
        "annotations": [],
    }

    img_id = 0
    ann_id = 0
    stats = {
        "images": 0,
        "images_with_instances": 0,
        "instances": 0,
        "skipped_no_mask": 0,
    }

    pbar = tqdm(images, desc=f"{split_name}", unit="img")
    for img_path in pbar:
        mask_path = masks_by_stem.get(img_path.stem)
        if mask_path is None:
            stats["skipped_no_mask"] += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            stats["skipped_no_mask"] += 1
            continue

        h, w = image.shape[:2]
        mask = _binarize(mask)
        if separate_touching:
            mask = _maybe_separate_touching(mask)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (w, h), interpolation=cv2.INTER_NEAREST,
            )

        instances = _instances_from_mask(mask, min_area=min_area)

        img_id += 1
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": int(w),
            "height": int(h),
        })
        stats["images"] += 1

        if instances:
            stats["images_with_instances"] += 1

        for (bbox_xywh, inst_mask) in instances:
            ann_id += 1
            rle = _encode_rle(inst_mask)
            area_px = int(inst_mask.sum())
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox_xywh,
                "area": area_px,
                "segmentation": rle,
                "iscrowd": 0,
            })
            stats["instances"] += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build COCO-style instance annotations from the existing binary "
            "masks (data/{split}/masks/)."
        )
    )
    parser.add_argument(
        "--min-area", type=int, default=config.INSTANCE_MIN_AREA_PX,
        help=(
            "Minimum number of pixels for a connected component to count as "
            f"a building instance (default {config.INSTANCE_MIN_AREA_PX})."
        ),
    )
    parser.add_argument(
        "--separate-touching", action="store_true",
        help=(
            "Apply a 3x3 erosion before labelling so that buildings sharing "
            "a wall in the GT mask split into separate components."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    splits = [
        ("train", config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR,
         config.TRAIN_INSTANCE_JSON),
        ("val",   config.VAL_IMAGES_DIR,   config.VAL_MASKS_DIR,
         config.VAL_INSTANCE_JSON),
        ("test",  config.TEST_IMAGES_DIR,  config.TEST_MASKS_DIR,
         config.TEST_INSTANCE_JSON),
    ]

    print("=" * 70)
    print("Building COCO instance annotations from binary masks")
    print("=" * 70)
    print(f"min_area           : {args.min_area} px")
    print(f"separate_touching  : {args.separate_touching}")
    print()

    summary = []
    for name, img_dir, mask_dir, out_json in splits:
        stats = _process_split(
            name, img_dir, mask_dir, out_json,
            min_area=args.min_area,
            separate_touching=args.separate_touching,
        )
        summary.append((name, stats, out_json))

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    header = (
        f"{'Split':<8}{'Images':>10}{'With inst.':>14}"
        f"{'Instances':>12}{'Avg/img':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, s, out_json in summary:
        avg = s["instances"] / s["images"] if s["images"] else 0.0
        with_pct = (
            s["images_with_instances"] / s["images"] * 100.0
            if s["images"] else 0.0
        )
        print(
            f"{name:<8}{s['images']:>10}"
            f"{s['images_with_instances']:>9} ({with_pct:>4.1f}%)"
            f"{s['instances']:>12}{avg:>10.2f}"
        )
    print("-" * len(header))
    print("\nCOCO JSON files written:")
    for _, _, out_json in summary:
        print(f"  {out_json}")
    print("\nDone. Next: python train_instance.py")


if __name__ == "__main__":
    main()
