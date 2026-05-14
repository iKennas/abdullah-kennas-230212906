"""Evaluate Mask R-CNN on the test set with COCO mAP and save report plots.

Outputs:
    reports/maskrcnn_map_bar.png        bbox AP and segm AP at standard IoU
                                        thresholds (AP, AP50, AP75)
    reports/maskrcnn_detections_grid.png example images with predicted boxes
                                        + masks + ground truth
    reports/maskrcnn_summary.txt        plain-text COCO metric table
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from build_model_instance import get_maskrcnn
from dataset_instance import (
    BuildingInstanceDataset,
    InstanceTransform,
    collate_fn,
)


REPORTS_DIR: Path = config.PROJECT_ROOT / "reports"


def load_checkpoint(ckpt_path: Path, device: str) -> torch.nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Mask R-CNN checkpoint not found: {ckpt_path}\n"
            "Run: python train_instance.py"
        )
    model = get_maskrcnn(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader,
                        device: str, score_thr: float
                        ) -> Tuple[List[Dict], List[Dict]]:
    """Return COCO-style detection and segmentation results lists."""
    det_results: List[Dict] = []
    seg_results: List[Dict] = []

    for images, targets in tqdm(loader, desc="Test inference", unit="batch"):
        images_on_dev = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images_on_dev)

        for tgt, out in zip(targets, outputs):
            image_id = int(tgt["image_id"].item())
            boxes  = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            masks  = (out["masks"].detach().cpu().numpy() > 0.5).astype(np.uint8)

            for i in range(boxes.shape[0]):
                if scores[i] < score_thr:
                    continue
                x1, y1, x2, y2 = boxes[i]
                w = float(x2 - x1)
                h = float(y2 - y1)
                if w < 1 or h < 1:
                    continue

                det_results.append({
                    "image_id":    image_id,
                    "category_id": int(labels[i]),
                    "bbox":        [float(x1), float(y1), w, h],
                    "score":       float(scores[i]),
                })

                inst_mask = masks[i, 0]
                rle = mask_utils.encode(np.asfortranarray(inst_mask))
                rle["counts"] = rle["counts"].decode("ascii")
                seg_results.append({
                    "image_id":     image_id,
                    "category_id":  int(labels[i]),
                    "segmentation": rle,
                    "score":        float(scores[i]),
                })

    return det_results, seg_results


def _coco_eval(coco_gt: COCO, coco_dt: COCO, iou_type: str) -> Dict[str, float]:
    """Run COCOeval with stdout silenced and return the 12 standard stats."""
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    keys = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    ]
    return dict(zip(keys, ev.stats.tolist()))


def evaluate_coco(gt_json: Path,
                  det_results: List[Dict], seg_results: List[Dict]
                  ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate both bbox and segmentation against the COCO ground truth."""
    coco_gt = COCO(str(gt_json))

    if not det_results:
        zero = {k: 0.0 for k in [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
        ]}
        return zero, dict(zero)

    coco_dt_bbox = coco_gt.loadRes(det_results)
    coco_dt_segm = coco_gt.loadRes(seg_results)

    bbox_stats = _coco_eval(coco_gt, coco_dt_bbox, "bbox")
    segm_stats = _coco_eval(coco_gt, coco_dt_segm, "segm")
    return bbox_stats, segm_stats


# ---------------------------------------------------------------------------
# Plotting / reporting
# ---------------------------------------------------------------------------

def plot_map_bar(bbox_stats: Dict[str, float], segm_stats: Dict[str, float],
                 out_path: Path) -> None:
    metrics = ["AP", "AP50", "AP75"]
    bbox_vals = [bbox_stats[m] for m in metrics]
    segm_vals = [segm_stats[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - width / 2, bbox_vals, width,
                label="bbox (detection)", color="tab:blue", alpha=0.85,
                edgecolor="black")
    b2 = ax.bar(x + width / 2, segm_vals, width,
                label="segm (instance mask)", color="tab:orange", alpha=0.85,
                edgecolor="black")
    for bars, vals in [(b1, bbox_vals), (b2, segm_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("COCO mAP")
    ax.set_title("Mask R-CNN - test-set mAP")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def _draw_detections(ax, image_rgb: np.ndarray,
                     boxes: np.ndarray, scores: np.ndarray,
                     masks: np.ndarray, title: str,
                     color: str = "lime") -> None:
    ax.imshow(image_rgb)
    ax.set_title(title)
    overlay = np.zeros_like(image_rgb, dtype=np.float32)
    for i in range(masks.shape[0]):
        m = masks[i]
        rgba = np.random.default_rng(i).random(3) * 0.7 + 0.3
        for c in range(3):
            overlay[..., c] += m * rgba[c] * 255
    blended = np.clip(image_rgb * 0.6 + overlay * 0.4, 0, 255).astype(np.uint8)
    ax.imshow(blended)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=1.2,
        )
        ax.add_patch(rect)
        if scores.size:
            ax.text(x1, y1 - 3, f"{scores[i]:.2f}",
                    color="white", fontsize=7,
                    bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"))
    ax.axis("off")


@torch.no_grad()
def plot_detection_grid(model: torch.nn.Module,
                        dataset: BuildingInstanceDataset,
                        device: str, indices: List[int],
                        score_thr: float, out_path: Path) -> None:
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5.0 * n))
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        image_tensor, target = dataset[idx]
        image_rgb = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_boxes = target["boxes"].numpy()
        gt_masks = target["masks"].numpy()

        _draw_detections(
            axes[row, 0], image_rgb, gt_boxes,
            np.zeros((0,), dtype=np.float32), gt_masks,
            title=f"Ground truth ({gt_boxes.shape[0]} buildings)",
            color="white",
        )

        out = model([image_tensor.to(device)])[0]
        boxes  = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        masks  = (out["masks"].cpu().numpy() > 0.5).astype(np.uint8)[:, 0]
        keep = scores >= score_thr
        boxes, scores, masks = boxes[keep], scores[keep], masks[keep]

        _draw_detections(
            axes[row, 1], image_rgb, boxes, scores, masks,
            title=f"Prediction ({boxes.shape[0]} buildings, score >= {score_thr})",
            color="lime",
        )

    fig.suptitle("Mask R-CNN - qualitative test examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def write_summary(bbox_stats: Dict[str, float], segm_stats: Dict[str, float],
                  out_path: Path) -> None:
    lines: List[str] = []
    header = f"{'Metric':<18}{'bbox AP':>12}{'segm AP':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]:
        lines.append(
            f"{k:<18}{bbox_stats[k]:>12.4f}{segm_stats[k]:>12.4f}"
        )
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print("\n" + text + "\n")
    print(f"  saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN on the test set.")
    parser.add_argument("--score-thr", type=float,
                        default=config.INSTANCE_SCORE_THRESHOLD,
                        help="Detection score threshold (default 0.5).")
    parser.add_argument("--num-grid", type=int, default=4,
                        help="Number of test images in the qualitative grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.set_seed()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Evaluating Mask R-CNN (instance segmentation)")
    print("=" * 70)
    print(f"  device     : {config.DEVICE}")
    print(f"  score thr  : {args.score_thr}")
    print(f"  reports dir: {REPORTS_DIR}")

    model = load_checkpoint(config.INSTANCE_CHECKPOINT, config.DEVICE)
    test_ds = BuildingInstanceDataset(
        config.TEST_IMAGES_DIR, config.TEST_INSTANCE_JSON,
        transforms=InstanceTransform(train=False),
        keep_empty=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.INSTANCE_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        collate_fn=collate_fn,
    )
    print(f"  test images: {len(test_ds)}\n")

    det_results, seg_results = collect_predictions(
        model, test_loader, config.DEVICE, args.score_thr,
    )
    print(f"  detections > {args.score_thr}: {len(det_results)}")

    bbox_stats, segm_stats = evaluate_coco(
        config.TEST_INSTANCE_JSON, det_results, seg_results,
    )

    print("\nSaving plots into reports/ ...")
    plot_map_bar(
        bbox_stats, segm_stats, REPORTS_DIR / "maskrcnn_map_bar.png",
    )

    rng = random.Random(config.SEED)
    candidate_ids = [
        i for i in range(len(test_ds))
        if test_ds.coco.getAnnIds(imgIds=test_ds.image_ids[i])
    ]
    if not candidate_ids:
        candidate_ids = list(range(len(test_ds)))
    indices = rng.sample(candidate_ids, k=min(args.num_grid, len(candidate_ids)))
    plot_detection_grid(
        model, test_ds, config.DEVICE, indices, args.score_thr,
        REPORTS_DIR / "maskrcnn_detections_grid.png",
    )

    write_summary(bbox_stats, segm_stats, REPORTS_DIR / "maskrcnn_summary.txt")

    raw_path = REPORTS_DIR / "maskrcnn_stats.json"
    raw_path.write_text(
        json.dumps({"bbox": bbox_stats, "segm": segm_stats}, indent=2),
        encoding="utf-8",
    )
    print(f"  saved {raw_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
