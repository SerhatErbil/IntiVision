import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from config import IMAGE_HEIGHT, IMAGE_WIDTH

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "intivision_v2_1.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"

CAMERA_INDEX = 0

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

HAND_PADDING_RATIO = 0.40
PREDICTION_THRESHOLD = 0.60


def load_labels():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            labels = json.load(file)

        if not isinstance(labels, dict):
            raise ValueError("labels.json must contain an index-label dictionary.")

        return labels

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"[ERROR] Labels could not be loaded: {error}")
        return None


def get_hand_box(frame, hand_landmarks):
    frame_height, frame_width = frame.shape[:2]

    x_coordinates = [
        int(landmark.x * frame_width)
        for landmark in hand_landmarks
    ]

    y_coordinates = [
        int(landmark.y * frame_height)
        for landmark in hand_landmarks
    ]

    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    hand_width = max_x - min_x
    hand_height = max_y - min_y

    hand_size = max(hand_width, hand_height)

    if hand_size <= 0:
        return None

    padding = int(hand_size * HAND_PADDING_RATIO)
    box_size = hand_size + (padding * 2)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    return x1, y1, x2, y2


def make_square_roi(frame, box):
    x1, y1, x2, y2 = box

    frame_height, frame_width = frame.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return None

    
    padding_left = max(0, -x1)
    padding_top = max(0, -y1)
    padding_right = max(0, x2 - frame_width)
    padding_bottom = max(0, y2 - frame_height)

    
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(frame_width, x2)
    crop_y2 = min(frame_height, y2)

    roi = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    if roi.size == 0:
        return None

    
    square_roi = cv2.copyMakeBorder(
        roi,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    
    roi_height, roi_width = square_roi.shape[:2]

    if roi_height != roi_width:
        square_size = max(roi_height, roi_width)

        extra_vertical = square_size - roi_height
        extra_horizontal = square_size - roi_width

        top = extra_vertical // 2
        bottom = extra_vertical - top
        left = extra_horizontal // 2
        right = extra_horizontal - left

        square_roi = cv2.copyMakeBorder(
            square_roi,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    return square_roi

def preprocess_roi(roi):
    resized_roi = cv2.resize(
        roi,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    rgb_roi = cv2.cvtColor(
        resized_roi,
        cv2.COLOR_BGR2RGB,
    )

    normalized_roi = rgb_roi.astype(np.float32) / 255.0

    model_input = np.expand_dims(
        normalized_roi,
        axis=0,
    )

    return model_input, resized_roi


def predict_gesture(model, labels, model_input):
    predictions = model.predict(
        model_input,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    predicted_label = labels.get(str(predicted_index))

    if predicted_label is None:
        raise ValueError(
            "Label not found for model output index: " f"{predicted_index}"
        )

    return predicted_label, confidence


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model could not be found: {MODEL_PATH}")
        return

    if not LABELS_PATH.exists():
        print(f"[ERROR] Labels could not be found: {LABELS_PATH}")
        return

    if not HAND_MODEL_PATH.exists():
        print("[ERROR] MediaPipe hand model could not be found:" f"\n{HAND_MODEL_PATH}")
        return

    labels = load_labels()

    if labels is None:
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except (OSError, ValueError) as error:
        print(f"[ERROR] Model could not be loaded: {error}")
        return

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Labels loaded: {labels}")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))

    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("[ERROR] Camera could not be opened.")
        return

    print("IntiVision V2.2 started.")
    print("Press 'q' to quit.")

    start_time = time.monotonic()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            hand_options
        ) as hand_landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("[ERROR] Frame could not be read.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int((time.monotonic() - start_time) * 1000)

                result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                prediction_text = "NO HAND"
                prediction_color = (0, 0, 255)

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]

                    hand_box = get_hand_box(
                        frame,
                        hand_landmarks,
                    )

                    if hand_box is not None:
                        x1, y1, x2, y2 = hand_box

                        square_roi = make_square_roi(
                            frame,
                            hand_box,
                        )

                        if square_roi is not None:
                            model_input, roi_preview = preprocess_roi(square_roi)

                            predicted_label, confidence = predict_gesture(
                                model,
                                labels,
                                model_input,
                            )

                            frame_height, frame_width = frame.shape[:2]

                            display_x1 = max(0, x1)
                            display_y1 = max(0, y1)
                            display_x2 = min(frame_width - 1, x2)
                            display_y2 = min(frame_height - 1, y2)

                            cv2.rectangle(
                                frame,
                                (display_x1, display_y1),
                                (display_x2, display_y2),
                                (0, 255, 0),
                                3,
                            )

                            prediction_text = (
                                f"{predicted_label.upper()} - "
                                f"{confidence * 100:.2f}%"
                            )

                            if confidence >= PREDICTION_THRESHOLD:
                                prediction_color = (0, 255, 0)
                            else:
                                prediction_color = (0, 165, 255)

                            cv2.imshow(
                                "Hand ROI - Model Input",
                                roi_preview,
                            )

                cv2.putText(
                    frame,
                    prediction_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    prediction_color,
                    2,
                )

                cv2.imshow(
                    "IntiVision V2.2",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as error:
        print(f"[ERROR] Realtime error: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()
