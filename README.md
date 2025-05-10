# Image Segmentation with U-Net

This project implements a U-Net-based deep learning model for semantic image segmentation using PyTorch. It processes images and annotations from the Stacked Carton Dataset, generates binary segmentation masks, trains a U-Net model, and performs inference to generate segmentation predictions. The pipeline includes data preparation, cleaning, training, and inference scripts, with support for GPU acceleration via Docker.

## Project Overview

The project is designed to:
- Convert COCO annotations to binary segmentation masks.
- Clean the dataset to remove mismatched or invalid image-mask pairs.
- Train a U-Net model for semantic segmentation using data augmentations.
- Perform inference using either a PyTorch checkpoint or an exported ONNX model.
- Evaluate model performance using pixel-wise accuracy and Dice score.

Key features:
- Modular codebase with separate scripts for data preparation, training, and inference.
- Support for mixed-precision training on CUDA-enabled GPUs.
- Data augmentation using Albumentations for robust training.
- Model export to ONNX format for deployment.
- TensorBoard integration for training visualization.

## Prerequisites

To run this project, ensure you have the following installed:

- **NVIDIA Drivers**: Required for GPU acceleration.
- **NVIDIA Container Toolkit**: For running the project in a Docker container with GPU support.
- **Docker**: To manage the containerized environment.
- Python 3.8+ (managed within the Docker container or installed locally for non-Docker setups).

### Tested Hardware
The project has been tested on the following configuration:
```
CPU: Ryzen 5 3600
RAM: 32 GB
GPU: NVIDIA RTX 5070Ti 16GB
```

## Setup

### Folder Structure
Before running the project, ensure your data directory is structured as follows:
```
data/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── images/
│   ├── train2017/
│   └── val2017/
```

After running the mask generation script, the structure should include mask directories:
```
data/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── images/
│   ├── train2017/
│   └── val2017/
├── masks/
│   ├── train2017/
│   └── val2017/
```

### Docker Setup
1. Pull or build the Docker image for the segmentation task (replace `segmentation_task` with your actual image name if different).
2. Start the Docker container with GPU support, mounting the project directory:
   ```bash
   docker run -it --gpus all --runtime nvidia --rm -v $(pwd):/home/seg-ws --name seg-container segmentation_task
   ```

This command:
- Runs the container interactively (`-it`).
- Enables all GPUs (`--gpus all`).
- Uses the NVIDIA runtime (`--runtime nvidia`).
- Removes the container after exit (`--rm`).
- Mounts the current directory to `/home/seg-ws` in the container (`-v $(pwd):/home/seg-ws`).

## Installation

### Using Docker
All dependencies are managed within the Docker container. Ensure the container image includes the packages listed in `requirements.txt`:
- `transformers==4.51.3`
- `datasets==3.5.1`
- `pycocotools==2.0.8`
- `albumentations==2.0.6`
- `numpy==2.2.5`
- `pillow==11.0.0`
- `scikit-learn==1.6.1`
- `matplotlib==3.10.1`
- `onnxruntime==1.21.1`
- `evaluate==0.4.3`
- `tensorboard==2.19.0`

If building the Docker image, include these dependencies in your `Dockerfile` or install them manually in the container using:
```bash
pip install -r requirements.txt
```

### Local Setup (Non-Docker)
For running the project without Docker, install the required packages locally:
1. Ensure Python 3.8+ is installed.
2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

Additionally, install PyTorch with CUDA support (if using a GPU) separately, as it is not included in `requirements.txt`. For example:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Adjust the CUDA version (`cu118`) based on your GPU and driver compatibility.

## Usage

Follow these steps to prepare data, clean the dataset, train the model, and perform inference.

### 1. Generate Masks
Convert COCO annotations to binary segmentation masks for training and validation datasets:
```bash
python src/utils/prepare_data.py
```
This script:
- Reads COCO annotations from `data/annotations/instances_train2017.json` and `data/annotations/instances_val2017.json`.
- Generates binary masks and saves them to `data/masks/train2017/` and `data/masks/val2017/`.
- Handles both polygon and RLE annotation formats, with error logging for problematic images.

### 2. Clean Dataset
Remove unmatched or mismatched image-mask pairs to ensure dataset integrity:
```bash
python src/utils/clean_data.py
```
This script:
- Compares files in `data/images/train2017/` with `data/masks/train2017/` and `data/images/val2017/` with `data/masks/val2017/`.
- Deletes files without corresponding pairs or with mismatched sizes.
- Prints a summary of deleted files.

### 3. Train the Model
Train the U-Net model using the prepared dataset:
```bash
python src/train.py
```
This script:
- Loads training and validation data using `DataLoader` and applies augmentations (horizontal flips, rotations, noise, blur for training; resizing and normalization for validation).
- Trains the U-Net model with the following hyperparameters (defined in `src

System: /config/params.py`):
  - Learning Rate: `5e-5`
  - Batch Size: `2`
  - Number of Epochs: `10`
  - Number of Workers: `1`
  - Image Size: `512x512`
- Uses BCEWithLogitsLoss and Adam optimizer.
- Saves checkpoints every 5 epochs to `outputs/checkpoints/`.
- Exports the best model (based on Dice score) to ONNX format at `outputs/best_model.onnx`.
- Logs training metrics to TensorBoard in `outputs/runs/`.

To visualize training progress, run TensorBoard in the container or locally:
```bash
tensorboard --logdir outputs/runs/
```

### 4. Perform Inference
Generate segmentation predictions for a test image:
```bash
python src/inference.py --model_type [checkpoint|onnx]
```
This script:
- Processes a test image (default: `data/images/test_images/test_img.jpg`).
- Supports two model types:
  - `--model_type checkpoint`: Uses a PyTorch checkpoint (e.g., `outputs/checkpoints/checkpoint_epoch_10.pth.tar`).
  - `--model_type onnx`: Uses the ONNX model (`outputs/best_model.onnx`).
- Applies validation augmentations (resize to 512x512, normalize).
- Saves the predicted mask to `outputs/pred_image1.png` and an upsampled version to `outputs/up_pred_image.png`.

Example:
```bash
python src/inference.py --model_type checkpoint
```

## Configuration

All paths and hyperparameters are defined in `src/config/params.py`:
- **Paths**:
  - `DATA_ROOT`: `data`
  - `TRAIN_IMG_DIR`: `data/images/train2017`
  - `TRAIN_MASK_DIR`: `data/masks/train2017`
  - `VAL_IMG_DIR`: `data/images/val2017`
  - `VAL_MASK_DIR`: `data/masks/val2017`
  - `CHECKPOINT_DIR`: `outputs/checkpoints/`
  - `OUTPUT_DIR`: `outputs/`
  - `ONNX_MODEL_PATH`: `outputs/best_model.onnx`
  - `IMAGE_PATH`: `data/images/test_images/test_img.jpg`
- **Hyperparameters**:
  - `LEARNING_RATE`: `5e-5`
  - `BATCH_SIZE`: `2`
  - `NUM_EPOCHS`: `10`
  - `NUM_WORKERS`: `1`
  - `IMAGE_HEIGHT`: `512`
  - `IMAGE_WIDTH`: `512`

Modify `params.py` to adjust paths or hyperparameters as needed.

## Project Structure

Key scripts and their purposes:
- `src/dataset.py`: Defines `SegmentationDataset` for loading image-mask pairs.
- `src/train.py`: Handles model training, checkpointing, and ONNX export.
- `src/inference.py`: Performs inference on test images using PyTorch or ONNX models.
- `src/utils/data_loader.py`: Creates `DataLoader` objects for training and validation.
- `src/utils/augmentation_policy.py`: Defines data augmentation pipelines using Albumentations.
- `src/utils/eval_metric.py`: Computes pixel-wise accuracy and Dice score for evaluation.
- `src/utils/get_device.py`: Selects the appropriate device (CUDA, MPS, or CPU).
- `src/utils/clean_data.py`: Cleans the dataset by removing unmatched or mismatched files.
- `src/utils/prepare_data.py`: Converts COCO annotations to binary masks.
- `src/config/params.py`: Central configuration file for paths and hyperparameters.
- `requirements.txt`: Lists Python package dependencies for the project.

## Notes

- Ensure the COCO dataset (images and annotations) is downloaded and placed in the `data/` directory before running `prepare_data.py`.
- The project assumes images are in `.jpg` format and masks are in `.png` format.
- For non-Docker setups, install dependencies using `requirements.txt` and ensure PyTorch is installed with appropriate CUDA support.
- For non-NVIDIA GPUs or CPU-only setups, modify `get_device.py` and remove NVIDIA-specific Docker flags.
- The ONNX model requires ONNX Runtime; ensure compatibility with your device (CUDA, CoreML, or CPU).

## Troubleshooting

- **Docker GPU issues**: Verify NVIDIA drivers and container toolkit are correctly installed. Check with `nvidia-smi`.
- **Missing annotations**: Ensure `instances_train2017.json` and `instances_val2017.json` are in `data/annotations/`.
- **Memory errors**: Reduce `BATCH_SIZE` in `params.py` or use a smaller image size.
- **ONNX inference errors**: Verify the ONNX model path and ensure ONNX Runtime supports your device.
- **Dependency issues**: Ensure all packages in `requirements.txt` are installed correctly, and check for version conflicts.

For further assistance, check the error logs printed by `prepare_data.py` or `clean_data.py`.