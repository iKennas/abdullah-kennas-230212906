"""Mask R-CNN factory (ResNet-50 + FPN backbone, COCO-pretrained).

``maskrcnn_resnet50_fpn_v2`` is the improved torchvision recipe (2022) that
includes a stronger backbone training schedule. We replace the box and mask
heads so they predict only ``background`` + ``building`` (2 classes).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import config


def get_maskrcnn(num_classes: int = config.INSTANCE_NUM_CLASSES,
                 pretrained: bool = True,
                 image_size: int = config.IMAGE_SIZE) -> nn.Module:
    """Mask R-CNN with the box and mask heads sized for ``num_classes``.

    By default torchvision rescales every input to ``min_size=800`` and
    ``max_size=1333`` inside the model. Our patches are 256x256 - upscaling
    them ~3x bloats memory and slows training without adding information.
    We override the internal transform to keep inputs at native size.

    Args:
        num_classes: number of classes including background.
        pretrained: load the official COCO-pretrained weights from torchvision.
        image_size: short-side resize used by the internal GeneralizedRCNN
            transform; both min_size and max_size are set to this value so
            patches keep their native resolution.

    Returns:
        A ``torchvision.models.detection.MaskRCNN`` module.
    """
    weights: Optional[str]
    weights_backbone: Optional[str]
    if pretrained:
        weights = "DEFAULT"
        weights_backbone = "DEFAULT"
    else:
        weights = None
        weights_backbone = "DEFAULT"  # keep ImageNet backbone init at least

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=weights_backbone,
        min_size=image_size, max_size=image_size,
    )

    # Replace the box classifier head.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor head.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Trainable weight count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = count_parameters(model)
    return (
        "Mask R-CNN (ResNet-50 + FPN, v2)\n"
        f"  Total parameters     : {n_total:>12,}  ({n_total / 1e6:6.2f} M)\n"
        f"  Trainable parameters : {n_trainable:>12,}  ({n_trainable / 1e6:6.2f} M)\n"
    )


if __name__ == "__main__":
    model = get_maskrcnn()
    print(model_summary(model))
    model.eval()
    with torch.no_grad():
        x = [torch.rand(3, config.IMAGE_SIZE, config.IMAGE_SIZE)]
        out = model(x)
    print(f"forward(1 image) -> output keys: {sorted(out[0].keys())}")
    for k, v in out[0].items():
        print(f"  {k:<8} shape={tuple(v.shape)}  dtype={v.dtype}")
