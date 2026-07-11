import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from config import IMAGE_HEIGHT, IMAGE_WIDTH, LABELS_PATH, MODEL_PATH


def load_labels():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[ERROR] Labels could not be loaded: {error}")
        return None


def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except (OSError, ValueError) as error:
        print(f"[ERROR] Model could not be loaded: {error}")
        return None


def preprocess_image(image_path):
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

            image_array = np.asarray(image, dtype=np.float32) / 255.0
            return np.expand_dims(image_array, axis=0)

    except (OSError, UnidentifiedImageError) as error:
        print(f"[ERROR] Image could not be processed: {error}")
        return None


def predict_image(image_path):
    model = load_model()
    labels = load_labels()

    if model is None or labels is None:
        return None

    image_array = preprocess_image(image_path)

    if image_array is None:
        return None

    predictions = model.predict(image_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    predicted_label = labels.get(str(predicted_index))

    if predicted_label is None:
        print(f"[ERROR] Label not found for model output index: {predicted_index}")
        return None

    return predicted_label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    result = predict_image(image_path)

    if result is None:
        sys.exit(1)

    label, confidence = result

    print("Prediction:", label)
    print(f"Confidence: {confidence * 100:.2f}%")