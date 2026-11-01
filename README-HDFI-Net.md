# HDFI-Net

HDFI-Net is a lightweight cotton disease detection model built on **Ultralytics YOLO11**. This repository provides the custom model architecture and scripts for training, validation, and test-set evaluation on custom datasets.

---

## Features

- **Layer-differentiated feature extraction** — DLB, GEB, and FCFB perform stage-specific feature processing.
- **Interactive downsampling** — DBID combines mixed pooling with parameter-free sequential spatial–channel bidirectional guidance.
- **Configurable experiments** — Dataset paths, output directories, devices, and training parameters can be configured by users.

---

## Requirements

### Reference environment

The following versions were recorded in the supplied Linux x86-64 environment:

| Component | Version / configuration |
|---|---|
| Python | 3.9.24 |
| PyTorch | 2.7.1+cu118 |
| Torchvision | 0.22.1+cu118 |
| PyTorch CUDA runtime | 11.8 |
| cuDNN | 9.1.0; reported runtime code: 90100 |
| Ultralytics source | 8.3.240, included in this repository |
| Reference GPU | NVIDIA A40 |

The GPU is a reference device, not a requirement to use the same model. GPU execution requires a compatible NVIDIA driver. The CUDA version displayed by `nvidia-smi` is the driver's supported CUDA version, not necessarily the runtime used by PyTorch.

### 1. Create a virtual environment

Replace `new_name` with your preferred environment name:

```bash
conda create -n new_name python=3.9.24 -y
conda activate new_name
python -m pip install pip==25.2
```

### 2. Install PyTorch

For NVIDIA GPU execution, install the CUDA 11.8 builds matching the recorded environment:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

The version pairing and installation index follow the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v271). Do not substitute the nightly or CUDA 12.1 command from another project's environment.

For CPU-only execution, use the following command **instead of** the CUDA command:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

The CPU-only option does not reproduce GPU throughput.

### 3. Install the required Python packages

The following direct dependencies cover the current training, validation, and testing entry points, including Albumentations used by the reference training environment:

```bash
python -m pip install numpy==2.0.2 scipy==1.13.0 matplotlib==3.9.4 pillow==10.4.0 PyYAML==6.0.1
python -m pip install requests==2.32.5 psutil==7.1.1 polars==1.34.0 ultralytics-thop==2.0.18 tqdm==4.67.1
python -m pip install opencv-python-headless==4.12.0.88 albumentations==2.0.8
python -m pip check
```

Headless OpenCV supports the image processing and saved plots used here without requiring an OpenCV GUI. Install only one OpenCV package variant; see the [OpenCV package guidance](https://pypi.org/project/opencv-python/4.12.0.88/).

These commands specify the relevant direct dependencies, not a complete lockfile for every transitive dependency. MMCV, MMEngine, Lightning, Torchaudio, ONNX Runtime, and Grad-CAM are not required for the three entry points above.

The modified `ultralytics/` package is bundled with this repository. A separate `pip install ultralytics` does not provide the HDFI-Net modules and is not needed for the workflow below. This copy has no `pyproject.toml`, so do not use `pip install -e .`.

---

## Installation & Usage

### 1. Download the repository

Download or clone this repository, then enter the directory containing `train.py`, `val.py`, and `test.py`:

```bash
cd HDFI-Net
```

Confirm that Python imports the bundled source rather than another installed copy:

```bash
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

The Ultralytics path must point inside this repository, and the source version should be `8.3.240`.

### 2. Prepare the dataset

Use YOLO detection annotations: one label file per image, with each object represented as `class_id x_center y_center width height`. Coordinates must be normalized to the image dimensions.

```text
cotton_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Edit `datasets/cotton.yaml` or provide your own dataset YAML. Fill in the dataset root and preserve the correct class order:

```yaml
path: ""  # Fill in the absolute dataset root.
train: train/images
val: val/images
test: test/images

names:
  0: BLI
  1: BS
  2: GS
  3: RL
  4: RSM
```

Dataset images and annotations are not bundled. To reproduce the paper's results, use the same data version, annotations, class order, and fixed train/validation/test split.

### 3. Train the model

Fill in the blank configuration fields at the top of `train.py`:

```python
MODEL_YAML_PATH = ""  # Path to ultralytics/cfg/models/HDFI-Net/HDFI-Net.yaml
DATA_YAML_PATH = ""
OUTPUT_PROJECT = ""
OUTPUT_NAME = ""
DEVICE = ""  # Automatic selection; use "0" for GPU 0 or "cpu" for CPU.
```

Then run:

```bash
python train.py
```

Default settings are training from scratch, up to **500 epochs**, **batch size 16**, **640 × 640** inputs, **AdamW**, an initial learning rate of **0.001**, weight decay of **0.0005**, AMP disabled, and early-stopping patience of **80**. The default random seed is **0**.

Parameters are defined in `TRAIN_ARGS`. Set `workers=0` when troubleshooting Windows data-loading issues. Reduce the batch size if the GPU has insufficient memory; record any changes when comparing results.

Outputs are written under `OUTPUT_PROJECT/OUTPUT_NAME/`. Use the checkpoint selected on the validation set, normally `weights/best.pt`, for final evaluation. Existing run directories are automatically incremented rather than overwritten.

For five-run experiments, repeat training from scratch with seeds **0–4**, using the same data split and settings and a separate output name for each run.

### 4. Validate the model

Fill in the following fields in `val.py`:

```python
WEIGHTS_PATH = ""  # Path to a trusted trained best.pt.
DATA_YAML_PATH = ""
OUTPUT_PROJECT = ""
OUTPUT_NAME = ""
DEVICE = ""
```

Run:

```bash
python val.py
```

This evaluates the **validation split**. Current defaults are `imgsz=640`, `batch=16`, `conf=0.001`, and `iou=0.70`. Other settings are defined in `VAL_ARGS`.

### 5. Evaluate on the test set

Fill in the same path and device fields separately in `test.py`, then run:

```bash
python test.py
```

This evaluates the **held-out test split**, with `imgsz=640`, `batch=4`, `conf=0.001`, and `iou=0.70`. Settings are defined in `TEST_ARGS`.

The script reports weight-file size, Precision, Recall, F1-score, mAP50, mAP50–95, and Pipeline FPS. F1 is calculated as the harmonic mean of the reported mean Precision and mean Recall. Pipeline FPS is calculated from preprocessing, inference, and postprocessing times; it excludes file loading, camera capture, GUI rendering, and network transfer.

The low confidence threshold is used for AP evaluation, and `iou` is the **NMS IoU threshold**, not the AP matching threshold. Keep evaluation settings consistent when comparing models. Timing results are specific to the hardware and software used.

All input and output fields are intentionally blank. Fill them in before execution; the scripts stop with a configuration error if required fields are missing. These entry points are configured in the files, not through `--data` or `--weights` command-line options.

---

## Project Structure

The HDFI-Net-specific files are organized as follows:

```text
HDFI-Net/
├── train.py
├── val.py
├── test.py
├── README-HDFI-Net.md
├── datasets/
│   └── cotton.yaml
├── checks/
│   ├── test_hdfi.py
│   └── smoke_train_val_test.py
└── ultralytics/
    ├── cfg/models/HDFI-Net/
    │   └── HDFI-Net.yaml
    └── nn/
        ├── tasks.py
        └── modules/
            ├── __init__.py
            ├── DLB.py
            ├── GEB.py
            ├── FCFB.py
            ├── DBID.py
            └── HDFI_common.py
```

`DLB`, `GEB`, and `FCFB` jointly implement LDFE. `DBID` implements interactive downsampling. The model YAML and newly trained checkpoints use these names. Checkpoints saved with the former research class names require separate compatibility handling before loading in this trimmed repository.

Optional functional checks:

```bash
python -m unittest discover -s checks -p test_hdfi.py -v
python checks/smoke_train_val_test.py
```

The second check creates temporary synthetic data for one CPU training epoch and validation/test evaluation, then removes the temporary outputs. These checks verify functionality, not reproduction of the paper's detection metrics.
