import json
import subprocess
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
STATUS_BAR_HEIGHT = 70
EMERGENCY_BLINK_INTERVAL = 0.45
ENABLE_EMERGENCY_SOUND = True
EMERGENCY_SOUND_PATH = (
    PROJECT_ROOT
    / "assets"
    / "sounds"
    / "ambulance_siren.mp3"
)

STATUS_COLORS = {
    "no_hand": (65, 65, 65),
    "analyzing": (0, 165, 255),
    "safe": (40, 170, 60),
    "not_safe": (0, 210, 255),
    "stop": (220, 120, 30),
    "help_code": (0, 120, 255),
    "emergency": (30, 30, 220),
}


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


def draw_status_bar(
    frame,
    status_key,
    status_text,
    confidence=None,
    is_stable=False,
):
    _, frame_width = frame.shape[:2]

    background_color = STATUS_COLORS.get(
        status_key,
        STATUS_COLORS["no_hand"],
    )

    text_color = (255, 255, 255)

    if status_key == "emergency" and is_stable:
        blink_phase = (
            int(time.monotonic() / EMERGENCY_BLINK_INTERVAL) % 2
        )

        if blink_phase == 0:
            background_color = STATUS_COLORS["emergency"]
            text_color = (255, 255, 255)
        else:
            background_color = (235, 235, 235)
            text_color = STATUS_COLORS["emergency"]

    cv2.rectangle(
        frame,
        (0, 0),
        (frame_width, STATUS_BAR_HEIGHT),
        background_color,
        thickness=-1,
    )

    display_text = status_text

    if confidence is not None:
        display_text += f"  {confidence * 100:.1f}%"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.05
    font_thickness = 2

    text_size, _ = cv2.getTextSize(
        display_text,
        font,
        font_scale,
        font_thickness,
    )

    text_width, text_height = text_size

    text_x = max(
        20,
        (frame_width - text_width) // 2,
    )

    text_y = (
        STATUS_BAR_HEIGHT
        + text_height
    ) // 2

    cv2.putText(
        frame,
        display_text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


def update_emergency_sound(is_active, sound_process):
    if not ENABLE_EMERGENCY_SOUND:
        return None

    if not EMERGENCY_SOUND_PATH.exists():
        return None

    if is_active:
        if sound_process is None or sound_process.poll() is not None:
            return subprocess.Popen(
                ["afplay", str(EMERGENCY_SOUND_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return sound_process

    if sound_process is not None and sound_process.poll() is None:
        sound_process.terminate()

        try:
            sound_process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            sound_process.kill()

    return None


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
    emergency_sound_process = None

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

                status_key = "no_hand"
                status_text = "NO HAND"
                status_confidence = None
                status_is_stable = False

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

                        status_key = predicted_label
                        status_text = predicted_label.replace("_", " ").upper()
                        status_confidence = confidence

                        if confidence >= PREDICTION_THRESHOLD:
                            current_time = time.monotonic()

                            if predicted_label != candidate_label:
                                candidate_label = predicted_label
                                candidate_started_at = current_time

                            stable_duration = (
                                current_time - candidate_started_at
                            )

                            prediction_is_stable = (
                                stable_duration
                                >= STABLE_PREDICTION_SECONDS
                            )
                            status_is_stable = prediction_is_stable

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
                            status_key = "analyzing"
                            status_text = "ANALYZING"
                            status_confidence = confidence
                            status_is_stable = False

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

                emergency_is_active = (
                    status_key == "emergency"
                    and status_is_stable
                )

                emergency_sound_process = update_emergency_sound(
                    emergency_is_active,
                    emergency_sound_process,
                )

                draw_status_bar(
                    frame=frame,
                    status_key=status_key,
                    status_text=status_text,
                    confidence=status_confidence,
                    is_stable=status_is_stable,
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
        update_emergency_sound(False, emergency_sound_process)
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()