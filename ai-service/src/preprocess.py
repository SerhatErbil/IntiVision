

import tensorflow as tf

from config import DATASET_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, GESTURE_CLASSES


BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42


def load_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=GESTURE_CLASSES,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=GESTURE_CLASSES,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        shuffle=True,
    )

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    validation_dataset = validation_dataset.map(
        lambda images, labels: (normalization_layer(images), labels)
    )

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset


if __name__ == "__main__":
    train_dataset, validation_dataset = load_datasets()

    print("Dataset loaded successfully.")
    print("Classes:", GESTURE_CLASSES)

    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Min pixel value:", tf.reduce_min(images).numpy())
        print("Max pixel value:", tf.reduce_max(images).numpy())