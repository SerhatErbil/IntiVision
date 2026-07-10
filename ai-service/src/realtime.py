import json
import time

import cv2
import numpy as np
import tensorflow as tf

from config import (
    CAMERA_INDEX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABELS_PATH,
    MODEL_PATH,
)
from event_client import send_prediction_event

def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

    frame_array = np.array(frame) / 255.0
    frame_array = np.expand_dims(frame_array, axis=0)

    return frame_array


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    last_detected_prediction = None
    last_detection_start_time = None
    last_sent_prediction = None

    STABLE_DURATION_SECONDS = 2.0
    EMERGENCY_STABLE_DURATION_SECONDS = 0.3
    CONFIDENCE_THRESHOLD = 0.80

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Frame could not be read.")
            break

        # Kamera görüntüsünü ayna gibi çeviriyoruz
        frame = cv2.flip(frame, 1)

        # ROI: Kameranın ortasında 300x300 el bölgesi oluşturuyoruz
        height, width, _ = frame.shape
        roi_size = 300

        x1 = width // 2 - roi_size // 2
        y1 = height // 2 - roi_size // 2
        x2 = width // 2 + roi_size // 2
        y2 = height // 2 + roi_size // 2

        # ROI: Sadece bu kutunun içindeki görüntüyü modele veriyoruz
        roi = frame[y1:y2, x1:x2]

        # Prediction artık tüm frame üzerinden değil, ROI üzerinden yapılıyor
        frame_array = preprocess_frame(roi)
        predictions = model.predict(frame_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        predicted_label = labels[str(predicted_index)]

        current_time = time.time()

        if confidence >= CONFIDENCE_THRESHOLD:
            if predicted_label != last_detected_prediction:
                last_detected_prediction = predicted_label
                last_detection_start_time = current_time

            stable_duration = current_time - last_detection_start_time

            required_duration = (
                EMERGENCY_STABLE_DURATION_SECONDS
                if predicted_label == "emergency"
                else STABLE_DURATION_SECONDS
            )

            if (
                stable_duration >= required_duration
                and predicted_label != last_sent_prediction
            ):
                event_sent = send_prediction_event(predicted_label, confidence)

                if event_sent:
                    last_sent_prediction = predicted_label
                    print(f"Event sent: {predicted_label} ({confidence * 100:.2f}%)")
        else:
             last_detected_prediction = None
             last_detection_start_time = None

        # UI renkleri
        if predicted_label == "safe":
            banner_color = (0, 180, 0)
            text_color = (255, 255, 255)
        elif predicted_label == "help_code":
            banner_color = (0, 215, 255)
            text_color = (0, 0, 0)
        elif predicted_label == "stop":
            banner_color = (0, 140, 255)
            text_color = (255, 255, 255)
        elif predicted_label == "not_safe":
            banner_color = (0, 0, 255)
            text_color = (255, 255, 255)
        elif predicted_label == "emergency":
            banner_color = (0, 0, 255)
            text_color = (255, 255, 255)
        else:
            banner_color = (70, 70, 70)
            text_color = (255, 255, 255)

        text = f"{predicted_label.upper()} - {confidence * 100:.2f}%"

        # ROI kutusunu ekrana çiziyoruz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Üstte durum bandı çiziyoruz
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 70),
            banner_color,
            -1,
        )

        # Tahmin sonucunu durum bandının üstüne yazıyoruz
        cv2.putText(
            frame,
            text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2,
        )

        cv2.imshow("IntiVision Realtime Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()