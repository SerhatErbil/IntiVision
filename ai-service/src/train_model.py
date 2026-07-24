from pathlib import Path

import tensorflow as tf

from config import (
    BASE_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    GESTURE_CLASSES,
)
from preprocess import load_datasets


MEDIAPIPE_DATASET_DIR = BASE_DIR / "dataset_mediapipe"

BEST_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "intivision_v2_2_best.keras"
)

FINAL_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "intivision_v2_2.keras"
)


def build_model():
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        )
    )

    model.add(
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
        )
    )

    model.add(
        tf.keras.layers.MaxPooling2D((2, 2))
    )

    model.add(
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
        )
    )

    model.add(
        tf.keras.layers.MaxPooling2D((2, 2))
    )

    model.add(
        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
        )
    )

    model.add(
        tf.keras.layers.MaxPooling2D((2, 2))
    )

    model.add(tf.keras.layers.Flatten())

    model.add(
        tf.keras.layers.Dense(
            128,
            activation="relu",
        )
    )

    model.add(tf.keras.layers.Dropout(0.5))

    model.add(
        tf.keras.layers.Dense(
            len(GESTURE_CLASSES),
            activation="softmax",
        )
    )

    return model


def main():
    train_dataset, validation_dataset = load_datasets(
        dataset_dir=MEDIAPIPE_DATASET_DIR
    )

    model = build_model()
    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=[
            early_stopping,
            checkpoint,
        ],
    )

    model.save(FINAL_MODEL_PATH)

    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Final model saved to: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()