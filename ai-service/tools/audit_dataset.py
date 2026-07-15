from collections import Counter
from pathlib import Path

from PIL import Image

from config import DATASET_DIR, GESTURE_CLASSES, IMAGE_WIDTH, IMAGE_HEIGHT


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(class_directory: Path) -> list[Path]:
    return [
        file_path
        for file_path in class_directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def audit_class(class_name: str) -> dict:
    class_directory = DATASET_DIR / class_name

    if not class_directory.exists():
        return {
            "class_name": class_name,
            "image_count": 0,
            "valid_count": 0,
            "corrupted_files": [],
            "image_sizes": Counter(),
            "missing_directory": True,
        }

    image_files = get_image_files(class_directory)

    corrupted_files = []
    image_sizes = Counter()
    valid_count = 0

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image_sizes[image.size] += 1

            valid_count += 1

        except (OSError, ValueError) as error:
            corrupted_files.append(
                {
                    "file": image_path.name,
                    "error": str(error),
                }
            )

    return {
        "class_name": class_name,
        "image_count": len(image_files),
        "valid_count": valid_count,
        "corrupted_files": corrupted_files,
        "image_sizes": image_sizes,
        "missing_directory": False,
    }


def main():
    print("=" * 60)
    print("INTIVISION DATASET AUDIT")
    print("=" * 60)
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Configured model input size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print()

    total_images = 0
    total_valid_images = 0
    total_corrupted_images = 0
    class_counts = {}

    for class_name in GESTURE_CLASSES:
        result = audit_class(class_name)

        print(f"Class: {class_name}")

        if result["missing_directory"]:
            print("  Directory does not exist.")
            print()
            continue

        image_count = result["image_count"]
        valid_count = result["valid_count"]
        corrupted_count = len(result["corrupted_files"])

        class_counts[class_name] = image_count

        total_images += image_count
        total_valid_images += valid_count
        total_corrupted_images += corrupted_count

        print(f"  Images: {image_count}")
        print(f"  Valid: {valid_count}")
        print(f"  Corrupted: {corrupted_count}")

        print("  Original image sizes:")

        for image_size, count in result["image_sizes"].most_common():
            width, height = image_size
            print(f"    {width}x{height}: {count}")

        if result["corrupted_files"]:
            print("  Corrupted files:")

            for corrupted_file in result["corrupted_files"]:
                print(
                    f"    {corrupted_file['file']}: "
                    f"{corrupted_file['error']}"
                )

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Valid images: {total_valid_images}")
    print(f"Corrupted images: {total_corrupted_images}")

    if total_images > 0:
        expected_train_count = int(total_images * 0.8)
        expected_validation_count = total_images - expected_train_count

        print()
        print("Approximate current split:")
        print(f"  Training: {expected_train_count} images (80%)")
        print(f"  Validation: {expected_validation_count} images (20%)")

    if class_counts:
        counts = list(class_counts.values())
        minimum_count = min(counts)
        maximum_count = max(counts)

        print()
        print("Class balance:")

        for class_name, count in class_counts.items():
            percentage = (
                count / total_images * 100
                if total_images > 0
                else 0
            )

            print(
                f"  {class_name}: "
                f"{count} images ({percentage:.2f}%)"
            )

        if minimum_count == maximum_count:
            print("  Dataset is perfectly balanced by image count.")
        else:
            print(
                "  Dataset is imbalanced. "
                f"Difference between largest and smallest class: "
                f"{maximum_count - minimum_count}"
            )


if __name__ == "__main__":
    main()