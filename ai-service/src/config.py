import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"


def resolve_path(environment_name: str, default_path: Path) -> Path:
    configured_path = os.getenv(environment_name)

    if not configured_path:
        return default_path

    path = Path(configured_path)

    if path.is_absolute():
        return path

    return BASE_DIR / path


MODEL_PATH = resolve_path(
    "MODEL_PATH",
    MODELS_DIR / "intivision_v1.keras",
)

LABELS_PATH = resolve_path(
    "LABELS_PATH",
    MODELS_DIR / "labels.json",
)

EVENT_API_URL = os.getenv(
    "EVENT_API_URL",
    "http://localhost:8080/api/v1/events",
)

DEVICE_ID = os.getenv("DEVICE_ID", "camera_01")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_COUNT_PER_CLASS = 150
IMAGE_BATCH_SIZE = 30

GESTURE_CLASSES = [
    "stop",
    "safe",
    "not_safe",
    "emergency",
    "help_code",
]