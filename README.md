Abdullah kennas 230212906

# Aerial Building Detection & Segmentation

**Formal course report (for grading):** the submitted write-up is the PDF
[**DERİN ÖĞRENME RAPORU ABDULLAH KENNAS.pdf**](./DER%C4%B0N%20%C3%96%C4%9ERENME%20RAPORU%20ABDULLAH%20KENNAS.pdf)
in the repository root (same folder as this `README.md`). This `README.md`
describes the code and how to run it; the PDF is the full narrative report.

Two complementary pipelines on the [Massachusetts Buildings
Dataset](https://www.cs.toronto.edu/~vmnih/data/):

1. **Semantic segmentation** — pixel-level binary masks with U-Net,
   U-Net++ and DeepLabV3+ (shared ResNet-34 encoder, ImageNet weights).
2. **Object detection + instance segmentation** — per-building bounding
   box, score and mask with **Mask R-CNN** (ResNet-50 + FPN, COCO weights).

Example prediction PNGs land in `outputs/predictions/` after you run
`evaluate.py` / `predict.py` (semantic) or `evaluate_instance.py` /
`predict_instance.py` (instance). Report figures land in `reports/`.

---

## 1. What this repo does

The three architectures all use a **ResNet-34 backbone with ImageNet
weights**, so differences mostly come from the decoder. Pipeline: tile the
official Kaggle splits into 256² patches → train with Albumentations → log
metrics and save checkpoints → compare models on the held-out test set.

Steps in code:

1. **`prepare_data.py`** – splits stay as provided; non-overlap tiling to 256.
2. **`dataset.py`** – PyTorch `Dataset`, Albumentations, encoder preprocessing.
3. **`build_model.py`** – [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch).
4. **`train.py`** – Dice+BCE, Adam, `ReduceLROnPlateau`, early stopping.
5. **`evaluate.py`** – IoU/Dice/etc. plus a few grids.
6. **`predict.py`** – sliding-window inference when the input is bigger than 256.

---

## 2. Dataset – Massachusetts Buildings

The [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/)
(Mnih, 2013) contains aerial imagery of the **Boston** area paired with
binary building masks.

| Property            | Value                              |
| ------------------- | ---------------------------------- |
| Number of images    | 151                                |
| Image size          | 1500 × 1500 px                     |
| Spatial resolution  | ≈ 1 m / pixel                      |
| Bands               | RGB                                |
| Mask type           | Binary (building = 1, else 0)      |
| Region              | Greater Boston metropolitan area   |

The dataset is also mirrored on [Kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset).

---

## 3. Model architectures

All three models share the **ResNet-34 encoder** (initialised with
ImageNet weights) — the differences below are **decoder-side**.

| Architecture     | Key idea                                                                                       | Reference                          |
| ---------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------- |
| **U-Net**        | Encoder–decoder with simple **skip connections** between matching levels.                      | Ronneberger et al., 2015           |
| **U-Net++**      | **Dense, nested** skip pathways close the semantic gap between low- and high-level features.   | Zhou et al., 2018                  |
| **DeepLabV3+**   | **Atrous (dilated) convolutions** + **ASPP** for multi-scale context, lightweight decoder.     | Chen et al., 2018                  |

---

## 4. Installation

```bash
# From the repo root (where train.py lives):
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> The `requirements.txt` pins **CPU-friendly** versions of every package.
> If you want GPU acceleration, install the appropriate CUDA build of
> PyTorch from https://pytorch.org/get-started/locally/ before running
> `pip install -r requirements.txt`.

---

## 5. Usage

### Step 1 – Download the dataset

Download the Massachusetts Buildings Dataset from
[Kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset).
The archive is already split:

```
massachusetts-buildings-dataset/
├── train/images/   train/masks/
├── val/images/     val/masks/
└── test/images/    test/masks/
```

(Optional) Set `RAW_DATA_DIR` in `config.py` to that folder so you can run
`prepare_data.py` without passing `--raw_dir` each time.

### Step 2 – Tile into patches (preserves Kaggle split)

```bash
python prepare_data.py --raw_dir "PATH_TO_EXTRACTED_FOLDER"
# Optional: --patch_size 256 (default matches config.IMAGE_SIZE)
```

This script:
- **Does not re-split** — train / val / test match Kaggle exactly.
- Crops each full-resolution image into **non-overlapping** patches (default
  **256×256**).
- Saves PNG patches under `data/train/`, `data/val/`, `data/test/`.
- Prints a summary including **% of patches that contain at least one building
  pixel**.

### Step 3 – Train each model

```bash
python train.py --model unet
python train.py --model unet++
python train.py --model deeplabv3+
```

Optional flags: `--epochs`, `--batch-size`, `--lr`.

Each run saves:
- `models/{name}_best.pth` (best val IoU checkpoint)
- `outputs/plots/{name}_training_curves.png`
- TensorBoard logs in `outputs/tensorboard/{name}/`

Launch TensorBoard with:

```bash
tensorboard --logdir outputs/tensorboard
```

### Step 4 – Evaluate on the test set

```bash
python evaluate.py --model unet
python evaluate.py --model unet++
python evaluate.py --model deeplabv3+
```

Produces `outputs/predictions/{model}_grid.png`,
`{model}_top5_best.png`, `{model}_top5_worst.png`,
`{model}_iou_histogram.png`.

### Step 5 – Predict on a brand-new image

```bash
python predict.py --image path/to/image.jpg --model unet
```

Or run **`python predict.py`** with no `--image`: it picks a random patch from
`data/test/images/` (useful right after data prep).

For a **custom** `--image` of any size, the script tiles with overlap, stitches
probabilities, and saves an overlay + mask under `outputs/predictions/`.

---

## 6. Results

After running `evaluate.py` on every model you will see per-model metrics
(IoU, Dice, pixel accuracy, precision, recall) plus parameter count and
average inference latency.

| Model      | Encoder  | Params (M) |
| ---------- | -------- | ---------- |
| unet       | resnet34 | 24.4       |
| unet++     | resnet34 | 26.1       |
| deeplabv3+ | resnet34 | 22.4       |

*(Fill in the metric columns after running the full pipeline on your
hardware.)*

Generated plots produced by `train.py` / `evaluate.py`:

| File                                                | Description                              |
| --------------------------------------------------- | ---------------------------------------- |
| `outputs/plots/{model}_training_curves.png`         | Loss / IoU / Dice per epoch              |
| `outputs/predictions/{model}_grid.png`              | Sample predictions on the test set       |
| `outputs/predictions/{model}_top5_best.png`         | Five best test patches by IoU            |
| `outputs/predictions/{model}_top5_worst.png`        | Five worst test patches by IoU           |
| `outputs/predictions/{model}_iou_histogram.png`     | Per-image IoU distribution               |

---

## 7. Project structure

```
derin/   (or whatever you name the submission folder)
├── data/
│   ├── raw/                <-- optional local mirror (prepare_data uses --raw_dir)
│   │   ├── images/
│   │   └── masks/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── models/                 <-- saved checkpoints (.pth)
├── outputs/
│   ├── predictions/        <-- evaluate.py / predict.py outputs
│   └── plots/              <-- training curves
├── utils/
│   ├── __init__.py
│   └── metrics.py          <-- IoU / Dice / Acc / Prec / Recall
├── reports/                <-- report-ready figures (IoU + Mask R-CNN mAP)
│
├── config.py               <-- single source of truth for hyper-params
├── prepare_data.py         <-- tile Kaggle train/val/test into patches (--raw_dir)
├── dataset.py              <-- PyTorch Dataset + Albumentations + DataLoaders
├── build_model.py          <-- factory for U-Net / U-Net++ / DeepLabV3+
├── train.py                <-- semantic training loop with early stopping
├── evaluate.py             <-- semantic test-set evaluation + visualisations
├── predict.py              <-- semantic inference on a single new image
│
├── prepare_instance_data.py    <-- masks -> COCO instance JSON
├── dataset_instance.py         <-- COCO-backed Dataset (boxes + masks)
├── build_model_instance.py     <-- Mask R-CNN factory
├── train_instance.py           <-- Mask R-CNN training loop
├── evaluate_instance.py        <-- COCO mAP + visualisations
├── predict_instance.py         <-- per-instance overlay PNG
│
├── DERİN ÖĞRENME RAPORU ABDULLAH KENNAS.pdf   <-- formal course report (PDF)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 8. References

- **segmentation_models_pytorch** — https://github.com/qubvel/segmentation_models.pytorch
- **Massachusetts Buildings Dataset** — Mnih, V. *Machine Learning for Aerial Image Labeling*, PhD thesis, University of Toronto, 2013. https://www.cs.toronto.edu/~vmnih/data/
- **U-Net** — Ronneberger, O., Fischer, P., & Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015. https://arxiv.org/abs/1505.04597
- **U-Net++** — Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* DLMIA, 2018. https://arxiv.org/abs/1807.10165
- **DeepLabV3+** — Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV, 2018. https://arxiv.org/abs/1802.02611
- **Albumentations** — https://albumentations.ai/
- **PyTorch** — https://pytorch.org/

---

## 9. Talking points / design notes

- Same ResNet-34 encoder everywhere so comparisons are decoder-focused.
- Dice + BCE: BCE trains pixel-wise gradients; Dice helps when backgrounds dominate.
- 1500² inputs are heavy on VRAM → fixed 256² patches during training.
- Kaggle splits are kept; patches from one raster never leak across splits.
- Early stopping when val loss stagnates even if train loss keeps improving.

---

## 10. Object detection + instance segmentation (Mask R-CNN)

This second pipeline turns the same dataset into an **object detection /
instance segmentation** problem: every building gets its own bounding box,
confidence score and per-instance mask, evaluated with **COCO mAP**.

### Pipeline

| Stage                | Script                          | What it does                                                                                  |
| -------------------- | ------------------------------- | --------------------------------------------------------------------------------------------- |
| Annotations          | `prepare_instance_data.py`      | `cv2.connectedComponentsWithStats` on each binary mask → COCO JSON with RLE per instance.     |
| Dataset              | `dataset_instance.py`           | PyTorch `Dataset` returning `(image, target_dict)` for torchvision detection models.          |
| Model                | `build_model_instance.py`       | `maskrcnn_resnet50_fpn_v2` (COCO-pretrained), box & mask heads resized to 2 classes.          |
| Training             | `train_instance.py`             | AdamW + StepLR, gradient clipping, 5-head loss sum, TensorBoard logs.                         |
| Evaluation           | `evaluate_instance.py`          | COCO `bbox` AP **and** `segm` AP via `pycocotools`, mAP bar chart + qualitative grid.         |
| Single-image predict | `predict_instance.py`           | Boxes + per-instance coloured masks + scores overlay PNG.                                     |

### Usage

```bash
pip install -r requirements.txt          # adds pycocotools
python prepare_instance_data.py          # data/*/instances.json  (uses existing masks)
python train_instance.py                 # --epochs 15  --batch-size 4
python evaluate_instance.py              # reports/maskrcnn_map_bar.png + grid + summary
python predict_instance.py               # random test patch, or --image PATH
```

### Key differences vs. the semantic pipeline

| Aspect            | U-Net / U-Net++ / DeepLabV3+         | Mask R-CNN                                |
| ----------------- | ------------------------------------ | ----------------------------------------- |
| Output            | One binary mask per image            | Per-object box + mask + score             |
| Encoder           | ResNet-34 (ImageNet)                 | ResNet-50 + FPN (COCO)                    |
| Loss              | Dice + BCE                           | Sum of 5 heads (RPN cls/box, RoI cls/box, mask) |
| Test metric       | IoU / Dice / pixel accuracy          | COCO mAP @ IoU 0.5 / 0.75 / 0.5–0.95      |
| Outputs           | `outputs/predictions/*_mask.png`     | `outputs/predictions/*_maskrcnn_pred.png` |
| Report files      | `reports/iou_*.png`                  | `reports/maskrcnn_*.png` + `maskrcnn_summary.txt` |

### Caveat: how instances are derived

Massachusetts Buildings ships **semantic** (binary) masks, not instance
masks. `prepare_instance_data.py` recovers instances via
`cv2.connectedComponentsWithStats`. Buildings sharing a wall therefore
collapse into one component. Pass `--separate-touching` to apply a 3×3
erosion before labelling and split some neighbours apart at the cost of
dropping the tiniest components.
