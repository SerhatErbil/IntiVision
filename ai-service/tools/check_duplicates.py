import hashlib
from collections import defaultdict
from pathlib import Path

from config import DATASET_DIR, GESTURE_CLASSES


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calculate_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_all_images() -> list[tuple[str, Path]]:
    images = []

    for class_name in GESTURE_CLASSES:
        class_directory = DATASET_DIR / class_name

        if not class_directory.exists():
            continue

        for file_path in sorted(class_directory.iterdir()):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                images.append((class_name, file_path))

    return images


def main():
    print("=" * 60)
    print("INTIVISION EXACT DUPLICATE CHECK")
    print("=" * 60)

    images = get_all_images()
    hash_groups = defaultdict(list)

    for class_name, image_path in images:
        file_hash = calculate_file_hash(image_path)
        hash_groups[file_hash].append((class_name, image_path.name))

    duplicate_groups = [
        group
        for group in hash_groups.values()
        if len(group) > 1
    ]

    print(f"Total images checked: {len(images)}")
    print(f"Exact duplicate groups: {len(duplicate_groups)}")

    duplicate_image_count = sum(
        len(group) for group in duplicate_groups
    )

    print(
        f"Images belonging to duplicate groups: "
        f"{duplicate_image_count}"
    )

    if not duplicate_groups:
        print("No exact duplicate files found.")
        return

    print()

    for index, group in enumerate(duplicate_groups, start=1):
        print(f"Duplicate group {index}:")

        for class_name, file_name in group:
            print(f"  {class_name}/{file_name}")

        print()


if __name__ == "__main__":
    main()