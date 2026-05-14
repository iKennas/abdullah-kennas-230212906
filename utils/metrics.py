"""IoU / Dice / acc / precision / recall for binary masks from logits."""

from __future__ import annotations

from typing import Dict

import torch


_SMOOTH: float = 1e-6


def _binarize(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Sigmoid + threshold; squeezes singleton channel."""
    if logits.dim() == 4 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target = (target > 0.5).float()
    return pred, target


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def iou_score(logits: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5) -> float:
    """Mean IoU."""
    pred, target = _binarize(logits, target, threshold)
    dims = (1, 2)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - intersection
    iou = (intersection + _SMOOTH) / (union + _SMOOTH)
    return float(iou.mean().item())


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> float:
    """Mean Dice."""
    pred, target = _binarize(logits, target, threshold)
    dims = (1, 2)
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + _SMOOTH) / (denom + _SMOOTH)
    return float(dice.mean().item())


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """Correct pixels / all pixels."""
    pred, target = _binarize(logits, target, threshold)
    correct = (pred == target).float()
    return float(correct.mean().item())


def precision_score(logits: torch.Tensor, target: torch.Tensor,
                    threshold: float = 0.5) -> float:
    """TP / (TP+FP)."""
    pred, target = _binarize(logits, target, threshold)
    dims = (1, 2)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    prec = (tp + _SMOOTH) / (tp + fp + _SMOOTH)
    return float(prec.mean().item())


def recall_score(logits: torch.Tensor, target: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """TP / (TP+FN)."""
    pred, target = _binarize(logits, target, threshold)
    dims = (1, 2)
    tp = (pred * target).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)
    rec = (tp + _SMOOTH) / (tp + fn + _SMOOTH)
    return float(rec.mean().item())


def compute_all_metrics(logits: torch.Tensor, target: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
    """All five floats in one dict."""
    return {
        "iou": iou_score(logits, target, threshold),
        "dice": dice_coefficient(logits, target, threshold),
        "accuracy": pixel_accuracy(logits, target, threshold),
        "precision": precision_score(logits, target, threshold),
        "recall": recall_score(logits, target, threshold),
    }
