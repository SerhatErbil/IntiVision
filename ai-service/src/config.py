from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"

CAMERA_INDEX = 0

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_COUNT_PER_CLASS = 30

GESTURE_CLASSES = [
    "open_hand",
    "fist",
    "peace",
    "thumbs_up",
    "thumbs_down",
]

IMAGE_BATCH_SIZE = 10