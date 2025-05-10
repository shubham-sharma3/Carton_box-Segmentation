import os

# Path Configurations
DATA_ROOT = "data"
ANN_PATH = "data/annotations"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images","train2017")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "masks","train2017")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "images","val2017")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "masks","val2017")
CHECKPOINT_DIR = "outputs/checkpoints/"
OUTPUT_DIR = "outputs/"
ONNX_MODEL_PATH = "outputs/best_model.onnx"
IMAGE_PATH = "data/images/test_images/test_img.jpg"


## Hyperparameters.
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
NUM_EPOCHS = 10
NUM_WORKERS = 1
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
