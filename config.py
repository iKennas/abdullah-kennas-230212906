"""Paths, hyperparameters and training defaults for the segmentation project."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Fix random / numpy / torch (and CUDA if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN gives reproducible results at a (small) speed cost.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
# DataLoader workers: 0 on Windows avoids multiprocessing quirks.
NUM_WORKERS: int = 0 if os.name == "nt" else 4


IMAGE_SIZE: int = 256          # patches are square IMAGE_SIZE x IMAGE_SIZE px
BATCH_SIZE: int = 16           # mini-batch size during training
NUM_EPOCHS: int = 30           # maximum number of training epochs
LEARNING_RATE: float = 1e-4    # Adam initial learning rate
EARLY_STOP_PATIENCE: int = 7   # epochs without val_loss improvement -> stop
LR_SCHEDULER_PATIENCE: int = 5 # epochs without val_loss improvement -> reduce LR


ENCODER_NAME: str = "resnet34"
ENCODER_WEIGHTS: str = "imagenet"
NUM_CLASSES: int = 1           # binary segmentation -> 1 logit per pixel
IN_CHANNELS: int = 3           # RGB satellite/aerial imagery
ACTIVATION = None              # None -> raw logits (BCEWithLogitsLoss handles σ)


PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Kaggle extract with train/val/test; or pass --raw_dir to prepare_data.py
RAW_DATA_DIR: Path = PROJECT_ROOT.parent / "massachusetts-buildings-dataset"


TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15


DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_IMAGES_DIR: Path = RAW_DIR / "images"
RAW_MASKS_DIR: Path = RAW_DIR / "masks"

# Processed patch folders (output of prepare_data.py)
TRAIN_IMAGES_DIR: Path = DATA_DIR / "train" / "images"
TRAIN_MASKS_DIR: Path = DATA_DIR / "train" / "masks"
VAL_IMAGES_DIR: Path = DATA_DIR / "val" / "images"
VAL_MASKS_DIR: Path = DATA_DIR / "val" / "masks"
TEST_IMAGES_DIR: Path = DATA_DIR / "test" / "images"
TEST_MASKS_DIR: Path = DATA_DIR / "test" / "masks"

# Aliases (same paths — some docs refer to *_IMG_DIR)
TRAIN_IMG_DIR: Path = TRAIN_IMAGES_DIR
TRAIN_MASK_DIR: Path = TRAIN_MASKS_DIR
VAL_IMG_DIR: Path = VAL_IMAGES_DIR
VAL_MASK_DIR: Path = VAL_MASKS_DIR
TEST_IMG_DIR: Path = TEST_IMAGES_DIR
TEST_MASK_DIR: Path = TEST_MASKS_DIR

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
TENSORBOARD_DIR: Path = OUTPUTS_DIR / "tensorboard"


def ensure_dirs() -> None:
    """Make sure data/model/output folders exist."""
    for path in [
        RAW_IMAGES_DIR, RAW_MASKS_DIR,
        TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR,
        VAL_IMAGES_DIR, VAL_MASKS_DIR,
        TEST_IMAGES_DIR, TEST_MASKS_DIR,
        MODELS_DIR, OUTPUTS_DIR, PREDICTIONS_DIR, PLOTS_DIR, TENSORBOARD_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Supported architectures (used by build_model.get_model and CLIs)
SUPPORTED_MODELS: tuple[str, ...] = ("unet", "unet++", "deeplabv3+")


# Path to the best checkpoint for a given model name -----------------------
def checkpoint_path(model_name: str) -> Path:
    """Best checkpoint for ``model_name`` (unet, unet++, deeplabv3+)."""
    safe = model_name.replace("+", "plus").replace("/", "_")
    return MODELS_DIR / f"{safe}_best.pth"


# ---------------------------------------------------------------------------
# Instance-segmentation (Mask R-CNN) configuration
# ---------------------------------------------------------------------------

# Number of detection classes (background + building).
INSTANCE_NUM_CLASSES: int = 2

# Class names indexed by class id; index 0 must stay "background".
INSTANCE_CLASS_NAMES: tuple[str, ...] = ("__background__", "building")

# A connected component must be at least this many pixels to count as an
# instance (filters mask noise).
INSTANCE_MIN_AREA_PX: int = 30

# Mask R-CNN is much heavier than U-Net, so use a smaller default batch.
INSTANCE_BATCH_SIZE: int = 4
INSTANCE_NUM_EPOCHS: int = 15
INSTANCE_LEARNING_RATE: float = 5e-4

# Detection confidence threshold used at evaluation and prediction time.
INSTANCE_SCORE_THRESHOLD: float = 0.5

# COCO-style annotation JSONs produced by prepare_instance_data.py.
TRAIN_INSTANCE_JSON: Path = DATA_DIR / "train" / "instances.json"
VAL_INSTANCE_JSON:   Path = DATA_DIR / "val"   / "instances.json"
TEST_INSTANCE_JSON:  Path = DATA_DIR / "test"  / "instances.json"

INSTANCE_CHECKPOINT: Path = MODELS_DIR / "maskrcnn_best.pth"


if __name__ == "__main__":
    set_seed()
    ensure_dirs()
    print("=" * 60)
    print("Aerial Building Segmentation - Configuration")
    print("=" * 60)
    print(f"Device              : {DEVICE}")
    print(f"Image size          : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Epochs (max)        : {NUM_EPOCHS}")
    print(f"Learning rate       : {LEARNING_RATE}")
    print(f"Encoder             : {ENCODER_NAME} ({ENCODER_WEIGHTS})")
    print(f"Raw data dir (hint) : {RAW_DATA_DIR}")
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Supported models    : {SUPPORTED_MODELS}")
    print("=" * 60)
