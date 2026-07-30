from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = AI_SERVICE_DIR.parent
SRC_DIR = AI_SERVICE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import GESTURE_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from hand_roi import DEFAULT_HAND_PADDING_RATIO, extract_hand_roi


V2_1_MODEL_PATH = (
    AI_SERVICE_DIR
    / "models"
    / "intivision_v2_1.keras"
)

V2_2_MODEL_PATH = (
    AI_SERVICE_DIR
    / "models"
    / "intivision_v2_2.keras"
)

HAND_MODEL_PATH = (
    AI_SERVICE_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

TEST_DATASET_DIR = PROJECT_DIR / "test_dataset"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def validate_paths() -> None:
    required_paths = [
        V2_1_MODEL_PATH,
        V2_2_MODEL_PATH,
        HAND_MODEL_PATH,
        TEST_DATASET_DIR,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path could not be found: {path}"
            )


def get_test_images() -> list[tuple[Path, int]]:
    test_images = []

    for class_index, class_name in enumerate(
        GESTURE_CLASSES
    ):
        class_dir = TEST_DATASET_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Test class directory not found: {class_dir}"
            )

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        for image_path in image_paths:
            test_images.append(
                (
                    image_path,
                    class_index,
                )
            )

    return test_images


def preprocess_v2_1(
    image: np.ndarray,
) -> np.ndarray:
    resized_image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    rgb_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB,
    )

    normalized_image = (
        rgb_image.astype(np.float32) / 255.0
    )

    return normalized_image


def preprocess_v2_2(
    image: np.ndarray,
    hand_landmarker,
) -> np.ndarray | None:
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image,
    )

    result = hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand_landmarks = result.hand_landmarks[0]

    roi_result = extract_hand_roi(
        frame=image,
        hand_landmarks=hand_landmarks,
        padding_ratio=DEFAULT_HAND_PADDING_RATIO,
    )

    if roi_result is None:
        return None

    square_roi, _ = roi_result

    resized_roi = cv2.resize(
        square_roi,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    rgb_roi = cv2.cvtColor(
        resized_roi,
        cv2.COLOR_BGR2RGB,
    )

    normalized_roi = (
        rgb_roi.astype(np.float32) / 255.0
    )

    return normalized_roi


def predict_classes(
    model,
    images: np.ndarray,
) -> np.ndarray:
    predictions = model.predict(
        images,
        verbose=0,
    )

    return np.argmax(
        predictions,
        axis=1,
    )

def save_confusion_matrix(
    title: str,
    file_name: str,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(GESTURE_CLASSES)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=GESTURE_CLASSES,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8),
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_dir = PROJECT_DIR / "docs" / "images"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / file_name

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Confusion matrix saved: {output_path}")

def print_results(
    title: str,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    correct = int(
        np.sum(true_labels == predicted_labels)
    )

    incorrect = int(
        np.sum(true_labels != predicted_labels)
    )

    accuracy = correct / len(true_labels)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Test samples  : {len(true_labels)}")
    print(f"Test accuracy : {accuracy:.4f}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {incorrect}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
            labels=range(len(GESTURE_CLASSES)),
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=range(len(GESTURE_CLASSES)),
            target_names=GESTURE_CLASSES,
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:
    validate_paths()

    test_images = get_test_images()

    print(f"Test images found: {len(test_images)}")

    v2_1_model = tf.keras.models.load_model(
        V2_1_MODEL_PATH
    )

    v2_2_model = tf.keras.models.load_model(
        V2_2_MODEL_PATH
    )

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    )

    hand_options = (
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.IMAGE
            ),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
    )

    v2_1_images = []
    v2_1_labels = []

    v2_2_images = []
    v2_2_labels = []

    common_v2_1_images = []
    common_labels = []

    rejected_images = []
    unreadable_images = []

    with mp.tasks.vision.HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker:

        for image_path, class_index in test_images:
            image = cv2.imread(str(image_path))

            if image is None:
                unreadable_images.append(image_path)
                continue

            v2_1_image = preprocess_v2_1(image)

            v2_1_images.append(v2_1_image)
            v2_1_labels.append(class_index)

            v2_2_image = preprocess_v2_2(
                image=image,
                hand_landmarker=hand_landmarker,
            )

            if v2_2_image is None:
                rejected_images.append(image_path)
                continue

            v2_2_images.append(v2_2_image)
            v2_2_labels.append(class_index)

            common_v2_1_images.append(v2_1_image)
            common_labels.append(class_index)

    if not v2_1_images:
        raise RuntimeError(
            "No readable V2.1 test images were found."
        )

    if not v2_2_images:
        raise RuntimeError(
            "MediaPipe could not process any test images."
        )

    v2_1_images_array = np.stack(v2_1_images)
    v2_1_labels_array = np.array(v2_1_labels)

    v2_2_images_array = np.stack(v2_2_images)
    v2_2_labels_array = np.array(v2_2_labels)

    common_v2_1_images_array = np.stack(
        common_v2_1_images
    )

    common_labels_array = np.array(common_labels)

    v2_1_predictions = predict_classes(
        model=v2_1_model,
        images=v2_1_images_array,
    )

    v2_2_predictions = predict_classes(
        model=v2_2_model,
        images=v2_2_images_array,
    )

    common_v2_1_predictions = predict_classes(
        model=v2_1_model,
        images=common_v2_1_images_array,
    )

    print_results(
        title="INTIVISION V2.1 — ORIGINAL TEST PIPELINE",
        true_labels=v2_1_labels_array,
        predicted_labels=v2_1_predictions,
    )

    print_results(
        title="INTIVISION V2.2 — MEDIAPIPE TEST PIPELINE",
        true_labels=v2_2_labels_array,
        predicted_labels=v2_2_predictions,
    )

    print_results(
        title="INTIVISION V2.1 — COMMON MEDIAPIPE-DETECTED SUBSET",
        true_labels=common_labels_array,
        predicted_labels=common_v2_1_predictions,
    )

    save_confusion_matrix(
        title="IntiVision V2.2 Confusion Matrix",
        file_name="confusion_matrix_v2_2.png",
        true_labels=v2_2_labels_array,
        predicted_labels=v2_2_predictions,
    )

    print("\n" + "=" * 70)
    print("MEDIAPIPE TEST PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total test images : {len(test_images)}")
    print(f"V2.2 processed    : {len(v2_2_images)}")
    print(f"Rejected          : {len(rejected_images)}")
    print(f"Unreadable        : {len(unreadable_images)}")

    if rejected_images:
        print("\nRejected images:")

        for image_path in rejected_images:
            print(f"- {image_path}")

    if unreadable_images:
        print("\nUnreadable images:")

        for image_path in unreadable_images:
            print(f"- {image_path}")


if __name__ == "__main__":
    main()