"""Instance-segmentation Dataset and DataLoader for Mask R-CNN.

Loads the COCO JSON written by ``prepare_instance_data.py`` and returns one
``(image_tensor, target_dict)`` pair per call, where ``target_dict`` has the
keys expected by ``torchvision.models.detection`` models:

    {
        "boxes":    (N, 4) float32 in [x1, y1, x2, y2] (absolute pixels),
        "labels":   (N,)   int64,
        "masks":    (N, H, W) uint8 {0, 1},
        "image_id": (1,)   int64,
        "area":     (N,)   float32,
        "iscrowd":  (N,)   int64,
    }

Random horizontal flip is the only augmentation: torchvision detection models
already include their own resize/normalisation transforms internally, and we
keep the augmentation budget light so building boxes stay tight.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

import config


# ---------------------------------------------------------------------------
# Augmentation (image + target)
# ---------------------------------------------------------------------------

def _hflip(image: np.ndarray, boxes: np.ndarray, masks: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal flip image, boxes and per-instance masks together."""
    w = image.shape[1]
    image = image[:, ::-1, :].copy()
    masks = masks[:, :, ::-1].copy()
    if boxes.size:
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
    return image, boxes, masks


class InstanceTransform:
    """Light augmentation pipeline that keeps image and target in sync."""

    def __init__(self, train: bool) -> None:
        self.train = train

    def __call__(self, image: np.ndarray, boxes: np.ndarray,
                 masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train and random.random() < 0.5:
            image, boxes, masks = _hflip(image, boxes, masks)
        return image, boxes, masks


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BuildingInstanceDataset(Dataset):
    """COCO-backed instance-segmentation dataset for Massachusetts Buildings."""

    def __init__(self, images_dir: Path, coco_json: Path,
                 transforms: Callable | None = None,
                 keep_empty: bool = False) -> None:
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(coco_json))
        self.transforms = transforms

        all_ids = list(self.coco.imgs.keys())
        if keep_empty:
            self.image_ids = all_ids
        else:
            # Drop patches with zero instances - Mask R-CNN trains faster
            # when every batch carries useful supervision signal.
            self.image_ids = [
                i for i in all_ids if self.coco.getAnnIds(imgIds=i)
            ]
        if not self.image_ids:
            raise RuntimeError(
                f"No usable images in {coco_json}. Did you run "
                "prepare_instance_data.py?"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, file_name: str) -> np.ndarray:
        img_path = self.images_dir / file_name
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _load_target(self, image_id: int, h: int, w: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read COCO annotations and return (boxes_xyxy, labels, masks)."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(int(ann["category_id"]))
            masks.append(self.coco.annToMask(ann).astype(np.uint8))

        if not boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, h, w), dtype=np.uint8),
            )
        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            np.stack(masks, axis=0).astype(np.uint8),
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        info = self.coco.imgs[image_id]
        image = self._load_image(info["file_name"])
        h, w = image.shape[:2]

        boxes, labels, masks = self._load_target(image_id, h, w)

        if self.transforms is not None:
            image, boxes, masks = self.transforms(image, boxes, masks)

        image_tensor = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float() / 255.0

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes_t.numel() == 0:
            boxes_t = boxes_t.reshape(0, 4)
        areas = (
            (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)
        )

        target: Dict[str, torch.Tensor] = {
            "boxes":    boxes_t,
            "labels":   torch.as_tensor(labels, dtype=torch.int64),
            "masks":    torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area":     areas.to(torch.float32),
            "iscrowd":  torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


# ---------------------------------------------------------------------------
# Collate + loader factory
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Torchvision detection models expect lists, not stacked tensors."""
    return tuple(zip(*batch))


def get_instance_loaders(batch_size: int = config.INSTANCE_BATCH_SIZE,
                         num_workers: int = config.NUM_WORKERS
                         ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val / test loaders for Mask R-CNN."""
    train_ds = BuildingInstanceDataset(
        config.TRAIN_IMAGES_DIR, config.TRAIN_INSTANCE_JSON,
        transforms=InstanceTransform(train=True),
        keep_empty=False,
    )
    val_ds = BuildingInstanceDataset(
        config.VAL_IMAGES_DIR, config.VAL_INSTANCE_JSON,
        transforms=InstanceTransform(train=False),
        keep_empty=True,
    )
    test_ds = BuildingInstanceDataset(
        config.TEST_IMAGES_DIR, config.TEST_INSTANCE_JSON,
        transforms=InstanceTransform(train=False),
        keep_empty=True,
    )

    pin = (config.DEVICE == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config.set_seed()
    train_loader, val_loader, test_loader = get_instance_loaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    images, targets = next(iter(train_loader))
    print(f"\nbatch len: {len(images)}")
    print(f"image[0] shape: {tuple(images[0].shape)} dtype={images[0].dtype}")
    print(f"target[0] keys: {sorted(targets[0].keys())}")
    print(f"target[0]['boxes'] shape: {tuple(targets[0]['boxes'].shape)}")
    print(f"target[0]['masks'] shape: {tuple(targets[0]['masks'].shape)}")
