from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

AI_SERVICE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = AI_SERVICE_DIR.parent
SRC_DIR = AI_SERVICE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import GESTURE_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH

MODEL_PATH = AI_SERVICE_DIR / "models" / "intivision_v2_1.keras"
TEST_DATASET_DIR = PROJECT_DIR / "test_dataset"
BATCH_SIZE = 32


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not TEST_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATASET_DIR}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=GESTURE_CLASSES,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    test_dataset = test_dataset.map(
        lambda images, labels: (
            normalization_layer(images),
            labels,
        )
    )

    test_dataset = test_dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    true_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        batch_predictions = np.argmax(predictions, axis=1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(batch_predictions)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    print("\n" + "=" * 60)
    print("INTIVISION INDEPENDENT TEST RESULTS")
    print("=" * 60)
    print(f"Test samples  : {len(true_labels)}")
    print(f"Test loss     : {loss:.6f}")
    print(f"Test accuracy : {accuracy:.4f}")
    print(
        f"Correct       : "
        f"{np.sum(true_labels == predicted_labels)}"
    )
    print(
        f"Incorrect     : "
        f"{np.sum(true_labels != predicted_labels)}"
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            predicted_labels,
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=GESTURE_CLASSES,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()