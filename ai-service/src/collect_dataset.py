import cv2
import time
from pathlib import Path

from config import (
    DATASET_DIR,
    CAMERA_INDEX,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_COUNT_PER_CLASS,
    GESTURE_CLASSES,
    IMAGE_BATCH_SIZE,
)


def create_dataset_folders():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for gesture_class in GESTURE_CLASSES:
        class_dir = DATASET_DIR / gesture_class
        class_dir.mkdir(parents=True, exist_ok=True)


def collect_images_for_class(class_name: str):
    class_dir = DATASET_DIR / class_name

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print(f"\nStarting data collection for: {class_name}")
    print("Press 's' to start capturing.")
    print("Press 'q' to quit.")

    started = False
    image_counter = 0

    while image_counter < IMAGE_COUNT_PER_CLASS:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        frame_height, frame_width, _ = frame.shape
        roi_size = 450
        roi_x = (frame_width - roi_size) // 2
        roi_y = (frame_height - roi_size) // 2

        roi = frame[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]

        cv2.rectangle(
            display_frame,
            (roi_x, roi_y),
            (roi_x + roi_size, roi_y + roi_size),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            "Place your hand inside the box",
            (roi_x - 30, roi_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Class: {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display_frame,
            f"Captured: {image_counter}/{IMAGE_COUNT_PER_CLASS}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if not started:
            cv2.putText(
                display_frame,
                "Press 's' to start",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

        cv2.imshow("IntiVision Dataset Collector", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            for countdown in range(3, 0, -1):
                print(countdown)
                time.sleep(1)

            started = True

        if started:
            resized_frame = cv2.resize(roi, (IMAGE_WIDTH, IMAGE_HEIGHT))

            image_path = class_dir / f"{class_name}_{image_counter:04d}.jpg"
            cv2.imwrite(str(image_path), resized_frame)

            image_counter += 1
            time.sleep(0.05)

        if image_counter % IMAGE_BATCH_SIZE == 0 and image_counter < IMAGE_COUNT_PER_CLASS:
            started = False
            print(f"\n{IMAGE_BATCH_SIZE} images captured.")
            print("Change hand position, angle or distance.")
            print("Press 's' to continue.")

        
    cap.release()
    cv2.destroyAllWindows()
    print(f"Completed collection for: {class_name}")


def main():
    create_dataset_folders()

    print("Available gesture classes:")

    for index, gesture_class in enumerate(GESTURE_CLASSES):
        print(f"{index + 1}. {gesture_class}")

    selected_index = int(input("\nSelect gesture class number: ")) - 1

    if selected_index < 0 or selected_index >= len(GESTURE_CLASSES):
        print("Invalid class selection.")
        return

    selected_class = GESTURE_CLASSES[selected_index]
    collect_images_for_class(selected_class)


if __name__ == "__main__":
    main()
