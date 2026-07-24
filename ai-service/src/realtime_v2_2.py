import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from config import IMAGE_HEIGHT, IMAGE_WIDTH
from event_client import send_prediction_event
from hand_roi import (
    DEFAULT_HAND_PADDING_RATIO,
    extract_hand_roi,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "intivision_v2_2.keras"
LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

CAMERA_INDEX = 0

MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

PREDICTION_THRESHOLD = 0.60
STABLE_PREDICTION_SECONDS = 1.0
NO_HAND_RESET_SECONDS = 1.0


def load_labels():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            labels = json.load(file)

        if not isinstance(labels, dict):
            raise ValueError(
                "labels.json must contain an index-label dictionary."
            )

        return labels

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"[ERROR] Labels could not be loaded: {error}")
        return None


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
            "Label not found for model output index: "
            f"{predicted_index}"
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
        print(
            "[ERROR] MediaPipe hand model could not be found:"
            f"\n{HAND_MODEL_PATH}"
        )
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

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    )

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

    last_sent_label = None

    candidate_label = None
    candidate_started_at = None

    no_hand_started_at = None

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

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                prediction_text = "NO HAND"
                prediction_color = (0, 0, 255)

                if result.hand_landmarks:
                    no_hand_started_at = None

                    hand_landmarks = result.hand_landmarks[0]

                    roi_result = extract_hand_roi(
                        frame=frame,
                        hand_landmarks=hand_landmarks,
                        padding_ratio=DEFAULT_HAND_PADDING_RATIO,
                    )

                    if roi_result is not None:
                        square_roi, hand_box = roi_result
                        x1, y1, x2, y2 = hand_box

                        model_input, roi_preview = preprocess_roi(
                            square_roi
                        )

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

                            current_time = time.monotonic()

                            if predicted_label != candidate_label:
                                candidate_label = predicted_label
                                candidate_started_at = current_time

                            stable_duration = (
                                current_time - candidate_started_at
                            )

                            prediction_text += (
                                f" | {stable_duration:.1f}/"
                                f"{STABLE_PREDICTION_SECONDS:.1f}s"
                            )

                            prediction_is_stable = (
                                stable_duration
                                >= STABLE_PREDICTION_SECONDS
                            )

                            should_send_event = (
                                prediction_is_stable
                                and predicted_label != last_sent_label
                            )

                            if should_send_event:
                                event_sent = send_prediction_event(
                                    predicted_label,
                                    confidence,
                                )

                                if event_sent:
                                    last_sent_label = predicted_label

                                    print(
                                        "[EVENT SENT] "
                                        f"gesture={predicted_label}, "
                                        f"confidence={confidence:.4f}, "
                                        f"stable_duration="
                                        f"{stable_duration:.2f}s"
                                    )

                        else:
                            prediction_color = (0, 165, 255)

                            candidate_label = None
                            candidate_started_at = None

                        cv2.imshow(
                            "Hand ROI - Model Input",
                            roi_preview,
                        )

                    else:
                        candidate_label = None
                        candidate_started_at = None

                else:
                    candidate_label = None
                    candidate_started_at = None

                    if no_hand_started_at is None:
                        no_hand_started_at = time.monotonic()

                    no_hand_duration = (
                        time.monotonic() - no_hand_started_at
                    )

                    if no_hand_duration >= NO_HAND_RESET_SECONDS:
                        last_sent_label = None

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