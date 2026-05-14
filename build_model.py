"""U-Net, U-Net++, DeepLabV3+ from segmentation_models_pytorch (shared ResNet-34)."""

from __future__ import annotations

from typing import Any, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

import config


_ARCHITECTURE_MAP: dict[str, Any] = {
    "unet":       smp.Unet,
    "unet++":     smp.UnetPlusPlus,
    "deeplabv3+": smp.DeepLabV3Plus,
}


def get_model(architecture: str = "unet",
              encoder: str = config.ENCODER_NAME,
              weights: Optional[str] = config.ENCODER_WEIGHTS,
              classes: int = config.NUM_CLASSES,
              in_channels: int = config.IN_CHANNELS,
              activation: Optional[str] = config.ACTIVATION,
              ) -> nn.Module:
    """Build SMP model: ``unet``, ``unet++``, or ``deeplabv3+``."""
    key = architecture.lower().strip()
    if key not in _ARCHITECTURE_MAP:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose one of {list(_ARCHITECTURE_MAP)}."
        )

    model_cls = _ARCHITECTURE_MAP[key]
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Trainable weights only."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, name: str = "") -> str:
    """Tiny text block with param counts (~M units)."""
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = count_parameters(model)
    header = f"Model summary{' - ' + name if name else ''}"
    return (
        f"{header}\n" + "-" * len(header) + "\n"
        f"  Total parameters     : {n_total:>12,}  ({n_total / 1e6:6.2f} M)\n"
        f"  Trainable parameters : {n_trainable:>12,}  ({n_trainable / 1e6:6.2f} M)\n"
    )


if __name__ == "__main__":
    for arch in config.SUPPORTED_MODELS:
        model = get_model(arch)
        print(model_summary(model, arch))

        x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            y = model(x)
        print(f"  forward({tuple(x.shape)}) -> {tuple(y.shape)}\n")
