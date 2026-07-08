from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"

CAMERA_INDEX = 0

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_COUNT_PER_CLASS = 200

GESTURE_CLASSES = [
    "stop",
    "safe",
    "not_safe",
    "emergency",
    "help_code",
]

IMAGE_BATCH_SIZE = 40