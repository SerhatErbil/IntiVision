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

def show_message_screen(cap, title, subtitle, footer, duration=2000):
    ret, frame = cap.read()

    if not ret:
        return

    frame = cv2.flip(frame, 1)

    cv2.putText(frame, title, (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(frame, subtitle, (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(frame, footer, (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("IntiVision Dataset Collector", frame)
    cv2.waitKey(duration)


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
    existing_images = list(class_dir.glob(f"{class_name}_*.jpg"))
    image_counter = len(existing_images)
    prev_time = time.time()
    fps = 0
    session_start_time = time.time()

    while image_counter < IMAGE_COUNT_PER_CLASS:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time > 0:
            fps = 1 / elapsed_time

        prev_time = current_time
        elapsed_time = int(current_time - session_start_time)

        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        frame_height, frame_width, _ = frame.shape
        roi_size = 450
        roi_x = (frame_width - roi_size) // 2
        roi_y = (frame_height - roi_size) // 2

        roi = frame[roi_y : roi_y + roi_size, roi_x : roi_x + roi_size]

        cv2.rectangle(
            display_frame,
            (roi_x, roi_y),
            (roi_x + roi_size, roi_y + roi_size),
            (0, 0, 255),
            4,
        )

        instruction_text = "Place your hand inside the box"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(
            instruction_text,
            font,
            font_scale,
            thickness,
        )

        text_x = roi_x + (roi_size - text_width) // 2
        text_y = roi_y - 20

        cv2.putText(
            display_frame,
            instruction_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
        )

        cv2.putText(
            display_frame,
            f"Class: {class_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"Captured: {image_counter}/{IMAGE_COUNT_PER_CLASS}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Elapsed: {minutes:02d}:{seconds:02d}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        progress_bar_x = 20
        progress_bar_y = 200
        progress_bar_width = 300
        progress_bar_height = 20

        progress_ratio = image_counter / IMAGE_COUNT_PER_CLASS
        filled_width = int(progress_bar_width * progress_ratio)

        cv2.rectangle(
            display_frame,
            (progress_bar_x, progress_bar_y),
            (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
            (255, 255, 255),
            2,
        )

        cv2.rectangle(
            display_frame,
            (progress_bar_x, progress_bar_y),
            (progress_bar_x + filled_width, progress_bar_y + progress_bar_height),
            (0, 255, 0),
            -1,
        )

        if not started:
            cv2.putText(
                display_frame,
                "Press 's' to start",
                (20, 250),
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
                ret, countdown_frame = cap.read()

                if not ret:
                    print("Failed to read frame during countdown.")
                    break

                countdown_frame = cv2.flip(countdown_frame, 1)
                countdown_display = countdown_frame.copy()

                cv2.rectangle(
                    countdown_display,
                    (roi_x, roi_y),
                    (roi_x + roi_size, roi_y + roi_size),
                    (0, 0, 255),
                    4,
                )

                cv2.putText(
                    countdown_display,
                    str(countdown),
                    (frame_width // 2 - 30, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    5,
                )

                cv2.imshow("IntiVision Dataset Collector", countdown_display)
                cv2.waitKey(1000)

            started = True

        if started:
            resized_frame = cv2.resize(roi, (IMAGE_WIDTH, IMAGE_HEIGHT))

            image_path = class_dir / f"{class_name}_{image_counter:04d}.jpg"
            cv2.imwrite(str(image_path), resized_frame)

            image_counter += 1
            time.sleep(0.05)

        if (
            started
            and image_counter > 0
            and image_counter % IMAGE_BATCH_SIZE == 0
            and image_counter < IMAGE_COUNT_PER_CLASS
        ):
            started = False
            print(f"\n{IMAGE_BATCH_SIZE} images captured.")
            print("Change hand position, angle or distance.")
            print("Press 's' to continue.")
            show_message_screen(
                cap,
                "Batch Complete",
                "Change hand position",
                "Press 'S' to continue",
            )

        if image_counter >= IMAGE_COUNT_PER_CLASS:
            show_message_screen(
                cap,
                "Dataset Completed",
                f"{class_name} finished",
                f"{image_counter} images saved",
                duration=3000,
            )
            
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
