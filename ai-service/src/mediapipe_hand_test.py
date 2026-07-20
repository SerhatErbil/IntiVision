import time
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAND_LANDMARKER_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

CAMERA_INDEX = 0

# MediaPipe el iskeletindeki landmark bağlantıları.
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


def draw_hand_landmarks(frame, hand_landmarks):
    frame_height, frame_width, _ = frame.shape
    landmark_points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        landmark_points.append((x, y))

    # Önce bağlantıları çiz.
    for start_index, end_index in HAND_CONNECTIONS:
        start_point = landmark_points[start_index]
        end_point = landmark_points[end_index]

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    # Ardından landmark noktalarını çiz.
    for point in landmark_points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 0, 255),
            -1,
        )


def draw_hand_box(frame, hand_landmarks, padding=25):
    frame_height, frame_width, _ = frame.shape

    x_coordinates = [
        int(landmark.x * frame_width)
        for landmark in hand_landmarks
    ]

    y_coordinates = [
        int(landmark.y * frame_height)
        for landmark in hand_landmarks
    ]

    x1 = max(min(x_coordinates) - padding, 0)
    y1 = max(min(y_coordinates) - padding, 0)

    x2 = min(max(x_coordinates) + padding, frame_width)
    y2 = min(max(y_coordinates) + padding, frame_height)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3,
    )

    return x1, y1, x2, y2


def main():
    if not HAND_LANDMARKER_MODEL_PATH.exists():
        print(
            "MediaPipe model could not be found:"
            f"\n{HAND_LANDMARKER_MODEL_PATH}"
        )
        return

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Camera could not be opened.")
        return

    print("Camera opened successfully.")
    print("Press 'q' to quit.")

    start_time = time.monotonic()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as hand_landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Frame could not be read.")
                    break

                # Kullanıcı ekranda kendisini ayna gibi görsün.
                frame = cv2.flip(frame, 1)

                # OpenCV BGR, MediaPipe SRGB/RGB bekliyor.
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

                if result.hand_landmarks:
                    cv2.putText(
                        frame,
                        "HAND DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    for hand_landmarks in result.hand_landmarks:
                        draw_hand_landmarks(
                            frame,
                            hand_landmarks,
                        )

                        draw_hand_box(
                            frame,
                            hand_landmarks,
                        )

                else:
                    cv2.putText(
                        frame,
                        "NO HAND",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(
                    "IntiVision - MediaPipe Hand Test",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as error:
        print(f"MediaPipe error: {error}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    main()