import csv
from pathlib import Path

import cv2
import mediapipe as mp

from config import DATASET_DIR, GESTURE_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
from hand_roi import DEFAULT_HAND_PADDING_RATIO, extract_hand_roi


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "dataset_mediapipe"
REPORT_PATH = PROJECT_ROOT / "mediapipe_processing_report.csv"

HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def create_output_folders() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in GESTURE_CLASSES:
        (OUTPUT_DIR / class_name).mkdir(
            parents=True,
            exist_ok=True,
        )


def resize_roi(roi):
    return cv2.resize(
        roi,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def main() -> None:
    if not HAND_MODEL_PATH.exists():
        print(
            "[ERROR] MediaPipe model could not be found:"
            f"\n{HAND_MODEL_PATH}"
        )
        return

    if not DATASET_DIR.exists():
        print(
            "[ERROR] Dataset directory could not be found:"
            f"\n{DATASET_DIR}"
        )
        return

    create_output_folders()

    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    )

    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    report_rows = []

    total_processed = 0
    total_success = 0
    total_rejected = 0
    total_errors = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        hand_options
    ) as hand_landmarker:

        for class_name in GESTURE_CLASSES:
            source_class_dir = DATASET_DIR / class_name
            output_class_dir = OUTPUT_DIR / class_name

            if not source_class_dir.exists():
                print(
                    f"[WARNING] Class folder not found: "
                    f"{source_class_dir}"
                )
                continue

            image_paths = get_image_paths(
                source_class_dir
            )

            class_success = 0
            class_rejected = 0
            class_errors = 0

            print(
                f"\nProcessing {class_name}: "
                f"{len(image_paths)} images"
            )

            for image_path in image_paths:
                total_processed += 1

                image = cv2.imread(str(image_path))

                if image is None:
                    total_errors += 1
                    class_errors += 1

                    report_rows.append(
                        {
                            "class_name": class_name,
                            "filename": image_path.name,
                            "status": "error",
                            "reason": "image_read_failed",
                        }
                    )

                    print(
                        f"[ERROR] Could not read: "
                        f"{image_path.name}"
                    )
                    continue

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image,
                )

                result = hand_landmarker.detect(
                    mp_image
                )

                if not result.hand_landmarks:
                    total_rejected += 1
                    class_rejected += 1

                    report_rows.append(
                        {
                            "class_name": class_name,
                            "filename": image_path.name,
                            "status": "rejected",
                            "reason": "no_hand_detected",
                        }
                    )

                    print(
                        f"[REJECTED] No hand: "
                        f"{image_path.name}"
                    )
                    continue

                hand_landmarks = result.hand_landmarks[0]

                roi_result = extract_hand_roi(
                    frame=image,
                    hand_landmarks=hand_landmarks,
                    padding_ratio=DEFAULT_HAND_PADDING_RATIO,
                )

                if roi_result is None:
                    total_rejected += 1
                    class_rejected += 1

                    report_rows.append(
                        {
                            "class_name": class_name,
                            "filename": image_path.name,
                            "status": "rejected",
                            "reason": "roi_creation_failed",
                        }
                    )

                    print(
                        f"[REJECTED] ROI failed: "
                        f"{image_path.name}"
                    )
                    continue

                square_roi, _ = roi_result
                resized_roi = resize_roi(square_roi)

                output_path = (
                    output_class_dir
                    / image_path.name
                )

                if output_path.exists():
                    print(
                        f"[SKIPPED] Already exists: "
                        f"{output_path.name}"
                    )
                    continue

                save_success = cv2.imwrite(
                    str(output_path),
                    resized_roi,
                )

                if not save_success:
                    total_errors += 1
                    class_errors += 1

                    report_rows.append(
                        {
                            "class_name": class_name,
                            "filename": image_path.name,
                            "status": "error",
                            "reason": "image_save_failed",
                        }
                    )

                    print(
                        f"[ERROR] Could not save: "
                        f"{output_path.name}"
                    )
                    continue

                total_success += 1
                class_success += 1

                report_rows.append(
                    {
                        "class_name": class_name,
                        "filename": image_path.name,
                        "status": "success",
                        "reason": "",
                    }
                )

                print(
                    f"[SUCCESS] {class_name}/"
                    f"{image_path.name}"
                )

            print(
                f"{class_name} summary -> "
                f"success: {class_success}, "
                f"rejected: {class_rejected}, "
                f"errors: {class_errors}"
            )

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "class_name",
                "filename",
                "status",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(report_rows)

    print("\n" + "=" * 50)
    print(f"Processed : {total_processed}")
    print(f"Success   : {total_success}")
    print(f"Rejected  : {total_rejected}")
    print(f"Errors    : {total_errors}")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Report    : {REPORT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()