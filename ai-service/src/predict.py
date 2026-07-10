import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image



from config import IMAGE_HEIGHT, IMAGE_WIDTH, LABELS_PATH, MODEL_PATH


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()

    image_array = preprocess_image(image_path)

    predictions = model.predict(image_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    predicted_label = labels[str(predicted_index)]

    return predicted_label, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    label, confidence = predict_image(image_path)

    print("Prediction:", label)
    print(f"Confidence: {confidence * 100:.2f}%")