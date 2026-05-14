"""Dataset / DataLoaders for Mass. Buildings patches from prepare_data.py."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

import config


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_training_augmentation(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """Flip/rotate, small affine + brightness jitters, dropout holes (train only)."""
    hole_max = max(image_size // 16, 1)
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5,
        ),
        A.GaussNoise(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(1, hole_max),
            hole_width_range=(1, hole_max),
            fill=0, fill_mask=0, p=0.3,
        ),
    ])


def get_validation_augmentation(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """Resize only (val/test)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
    ])


def get_preprocessing(preprocessing_fn: Callable) -> A.Compose:
    """SMP encoder mean/std normalize + tensors (float32)."""
    def _preprocess_image(image: np.ndarray, **kwargs) -> np.ndarray:
        return preprocessing_fn(image).astype(np.float32)

    def _preprocess_mask(mask: np.ndarray, **kwargs) -> np.ndarray:
        return (mask > 0).astype(np.float32)

    return A.Compose([
        A.Lambda(image=_preprocess_image, mask=_preprocess_mask),
        ToTensorV2(),
    ])


class BuildingDataset(Dataset):
    """Returns normalized image tensor (3,H,W) and mask (1,H,W) float {0,1}."""

    _IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, images_dir: Path, masks_dir: Path,
                 augmentation: Optional[A.Compose] = None,
                 preprocessing: Optional[A.Compose] = None) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.image_paths: List[Path] = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self._IMG_EXTENSIONS
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.images_dir}. "
                "Did you run prepare_data.py?"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resolve_mask_path(self, img_path: Path) -> Path:
        """Same name as image, or same stem with another raster ext."""
        exact = self.masks_dir / img_path.name
        if exact.exists():
            return exact
        stem = img_path.stem
        for ext in self._IMG_EXTENSIONS:
            candidate = self.masks_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No mask found for image '{img_path.name}' (stem '{stem}') "
            f"in {self.masks_dir}"
        )

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load one RGB patch and mask."""
        img_path = self.image_paths[idx]
        mask_path = self._resolve_mask_path(img_path)

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load_pair(idx)

        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy((mask > 0).astype(np.float32))

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        return image, mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_loaders(batch_size: int = config.BATCH_SIZE,
                num_workers: int = config.NUM_WORKERS,
                encoder_name: str = config.ENCODER_NAME,
                encoder_weights: str = config.ENCODER_WEIGHTS,
                ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val / test loaders with encoder preprocessing."""
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        encoder_name, pretrained=encoder_weights,
    )
    preprocessing = get_preprocessing(preprocessing_fn)

    train_ds = BuildingDataset(
        config.TRAIN_IMAGES_DIR, config.TRAIN_MASKS_DIR,
        augmentation=get_training_augmentation(),
        preprocessing=preprocessing,
    )
    val_ds = BuildingDataset(
        config.VAL_IMAGES_DIR, config.VAL_MASKS_DIR,
        augmentation=get_validation_augmentation(),
        preprocessing=preprocessing,
    )
    test_ds = BuildingDataset(
        config.TEST_IMAGES_DIR, config.TEST_MASKS_DIR,
        augmentation=get_validation_augmentation(),
        preprocessing=preprocessing,
    )

    pin = (config.DEVICE == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config.set_seed()
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    images, masks = next(iter(train_loader))
    print(f"image batch shape: {tuple(images.shape)} dtype={images.dtype}")
    print(f"mask  batch shape: {tuple(masks.shape)} dtype={masks.dtype}")
    print(f"mask  unique values: {torch.unique(masks).tolist()}")
