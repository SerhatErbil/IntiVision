
import re
import time
from pathlib import Path

import cv2

from config import (
    DATASET_DIR,
    CAMERA_INDEX,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    GESTURE_CLASSES,
    IMAGE_BATCH_SIZE,
)


WINDOW_NAME = "IntiVision Dataset Collector"
DEFAULT_BATCH_SIZE = 20


def create_dataset_folders() -> None:
    """Create the dataset root and class folders if they do not exist."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for gesture_class in GESTURE_CLASSES:
        class_dir = DATASET_DIR / gesture_class
        class_dir.mkdir(parents=True, exist_ok=True)


def get_existing_images(class_dir: Path, class_name: str) -> list[Path]:
    """Return existing JPG images belonging to the selected class."""
    return sorted(class_dir.glob(f"{class_name}_*.jpg"))


def get_next_image_index(class_dir: Path, class_name: str) -> int:
    """
    Find the highest numeric suffix in existing filenames.

    Example:
        stop_0199.jpg -> next index is 200

    File count is intentionally not used because files may have been deleted.
    """
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$")
    highest_index = -1

    for image_path in class_dir.glob(f"{class_name}_*.jpg"):
        match = pattern.match(image_path.name)

        if not match:
            continue

        image_index = int(match.group(1))
        highest_index = max(highest_index, image_index)

    return highest_index + 1


def get_positive_integer(
    prompt: str,
    default: int | None = None,
) -> int:
    """Read a positive integer from the terminal."""
    while True:
        user_input = input(prompt).strip()

        if not user_input and default is not None:
            return default

        try:
            value = int(user_input)

            if value <= 0:
                print("Please enter a number greater than 0.")
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def show_message_screen(
    cap: cv2.VideoCapture,
    title: str,
    subtitle: str,
    footer: str,
    duration: int = 2000,
) -> None:
    """Display a temporary information screen on the camera window."""
    ret, frame = cap.read()

    if not ret:
        return

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        title,
        (80, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        4,
    )

    cv2.putText(
        frame,
        subtitle,
        (80, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        3,
    )

    cv2.putText(
        frame,
        footer,
        (80, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(duration)


def wait_for_batch_continue(
    cap: cv2.VideoCapture,
    class_name: str,
    session_captured: int,
    images_to_add: int,
) -> bool:
    """
    Pause after a batch.

    Returns:
        True if the user wants to continue.
        False if the user quits.
    """
    print(f"\nBatch completed: {session_captured}/{images_to_add}")
    print("Change hand position, angle, distance or wrist rotation.")
    print("Press 's' to continue or 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame while waiting between batches.")
            return False

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "Batch Complete",
            (70, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            4,
        )

        cv2.putText(
            frame,
            "Change hand position, angle or distance",
            (70, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"{class_name}: {session_captured}/{images_to_add}",
            (70, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Press 's' to continue or 'q' to quit",
            (70, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            return True

        if key == ord("q"):
            return False


def run_countdown(
    cap: cv2.VideoCapture,
    roi_x: int,
    roi_y: int,
    roi_size: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Display a three-second countdown before capturing."""
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame during countdown.")
            return False

        frame = cv2.flip(frame, 1)
        countdown_display = frame.copy()

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
            (frame_width // 2 - 35, frame_height // 2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            5,
        )

        cv2.imshow(WINDOW_NAME, countdown_display)

        key = cv2.waitKey(1000) & 0xFF

        if key == ord("q"):
            return False

    return True


def collect_images_for_class(
    class_name: str,
    images_to_add: int,
    batch_size: int,
) -> None:
    """Collect a requested number of new images for one gesture class."""
    class_dir = DATASET_DIR / class_name
    existing_images = get_existing_images(class_dir, class_name)

    current_image_count = len(existing_images)
    next_image_index = get_next_image_index(class_dir, class_name)
    target_total = current_image_count + images_to_add

    print("\n" + "=" * 50)
    print(f"Class          : {class_name}")
    print(f"Current images : {current_image_count}")
    print(f"Images to add  : {images_to_add}")
    print(f"Target total   : {target_total}")
    print(f"Next filename  : {class_name}_{next_image_index:04d}.jpg")
    print(f"Batch size     : {batch_size}")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    print("\nPress 's' to start capturing.")
    print("Press 'q' to quit.")

    started = False
    session_captured = 0
    prev_time = time.time()
    session_start_time = time.time()
    fps = 0.0

    try:
        while session_captured < images_to_add:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            current_time = time.time()
            frame_duration = current_time - prev_time

            if frame_duration > 0:
                fps = 1 / frame_duration

            prev_time = current_time

            elapsed_seconds = int(current_time - session_start_time)
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60

            frame_height, frame_width, _ = frame.shape

            roi_size = min(450, frame_width, frame_height)
            roi_x = (frame_width - roi_size) // 2
            roi_y = (frame_height - roi_size) // 2

            roi = frame[
                roi_y : roi_y + roi_size,
                roi_x : roi_x + roi_size,
            ]

            if roi.size == 0:
                print("ROI could not be created from the camera frame.")
                break

            cv2.rectangle(
                display_frame,
                (roi_x, roi_y),
                (roi_x + roi_size, roi_y + roi_size),
                (0, 0, 255),
                4,
            )

            instruction_text = "Place your hand inside the box"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (text_width, _), _ = cv2.getTextSize(
                instruction_text,
                font,
                font_scale,
                thickness,
            )

            text_x = roi_x + (roi_size - text_width) // 2
            text_y = max(30, roi_y - 20)

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
                0.9,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Session: {session_captured}/{images_to_add}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Dataset total: {current_image_count + session_captured}/{target_total}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display_frame,
                f"Elapsed: {minutes:02d}:{seconds:02d}",
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            progress_bar_x = 20
            progress_bar_y = 220
            progress_bar_width = 300
            progress_bar_height = 20

            progress_ratio = session_captured / images_to_add
            filled_width = int(progress_bar_width * progress_ratio)

            cv2.rectangle(
                display_frame,
                (progress_bar_x, progress_bar_y),
                (
                    progress_bar_x + progress_bar_width,
                    progress_bar_y + progress_bar_height,
                ),
                (255, 255, 255),
                2,
            )

            cv2.rectangle(
                display_frame,
                (progress_bar_x, progress_bar_y),
                (
                    progress_bar_x + filled_width,
                    progress_bar_y + progress_bar_height,
                ),
                (0, 255, 0),
                -1,
            )

            if not started:
                cv2.putText(
                    display_frame,
                    "Press 's' to start",
                    (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Collection stopped by user.")
                break

            if key == ord("s") and not started:
                countdown_completed = run_countdown(
                    cap=cap,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    roi_size=roi_size,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                if not countdown_completed:
                    print("Collection cancelled during countdown.")
                    break

                started = True
                session_start_time = time.time()
                continue

            if not started:
                continue

            resized_frame = cv2.resize(
                roi,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
            )

            image_path = (
                class_dir
                / f"{class_name}_{next_image_index:04d}.jpg"
            )

            if image_path.exists():
                print(
                    f"Safety stop: file already exists and will not be overwritten: "
                    f"{image_path.name}"
                )
                break

            save_success = cv2.imwrite(
                str(image_path),
                resized_frame,
            )

            if not save_success:
                print(f"Failed to save image: {image_path}")
                break

            next_image_index += 1
            session_captured += 1

            time.sleep(0.05)

            batch_completed = (
                session_captured % batch_size == 0
                and session_captured < images_to_add
            )

            if batch_completed:
                started = False

                should_continue = wait_for_batch_continue(
                    cap=cap,
                    class_name=class_name,
                    session_captured=session_captured,
                    images_to_add=images_to_add,
                )

                if not should_continue:
                    print("Collection stopped between batches.")
                    break

                countdown_completed = run_countdown(
                    cap=cap,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    roi_size=roi_size,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                if not countdown_completed:
                    print("Collection cancelled during countdown.")
                    break

                started = True

        final_total = len(get_existing_images(class_dir, class_name))

        if session_captured == images_to_add:
            show_message_screen(
                cap,
                "Session Completed",
                f"{class_name}: {session_captured} new images",
                f"Dataset total: {final_total}",
                duration=3000,
            )

        print("\n" + "=" * 50)
        print(f"Collection finished for : {class_name}")
        print(f"New images saved        : {session_captured}")
        print(f"Previous total          : {current_image_count}")
        print(f"Current total           : {final_total}")
        print("=" * 50)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    create_dataset_folders()

    print("\nAvailable gesture classes:")

    for index, gesture_class in enumerate(GESTURE_CLASSES, start=1):
        print(f"{index} - {gesture_class}")

    try:
        selected_index = int(
            input("\nSelect gesture class number: ").strip()
        ) - 1

    except ValueError:
        print("Invalid class selection. Please enter a number.")
        return

    if selected_index < 0 or selected_index >= len(GESTURE_CLASSES):
        print("Invalid class selection.")
        return

    selected_class = GESTURE_CLASSES[selected_index]
    class_dir = DATASET_DIR / selected_class

    current_image_count = len(
        get_existing_images(class_dir, selected_class)
    )

    print(f"\nSelected class : {selected_class}")
    print(f"Current images : {current_image_count}")

    images_to_add = get_positive_integer(
        "How many new images should be added in this session? "
    )

    batch_size = get_positive_integer(
        f"Batch size [default {DEFAULT_BATCH_SIZE}]: ",
        default=DEFAULT_BATCH_SIZE,
    )

    target_total = current_image_count + images_to_add

    print("\nCollection summary:")
    print(f"Current images : {current_image_count}")
    print(f"Images to add  : {images_to_add}")
    print(f"Target total   : {target_total}")
    print(f"Batch size     : {batch_size}")

    confirmation = input(
        "\nStart this collection session? [y/N]: "
    ).strip().lower()

    if confirmation != "y":
        print("Collection cancelled.")
        return

    collect_images_for_class(
        class_name=selected_class,
        images_to_add=images_to_add,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()

