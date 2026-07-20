import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from config import DATASET_DIR, GESTURE_CLASSES

IMAGES_PER_CLASS = 4

fig, axes = plt.subplots(
    len(GESTURE_CLASSES),
    IMAGES_PER_CLASS,
    figsize=(12, 15),
)

fig.suptitle("IntiVision Dataset Preview", fontsize=18)

for row, class_name in enumerate(GESTURE_CLASSES):
    class_dir = DATASET_DIR / class_name

    images = list(class_dir.glob("*.jpg"))

    selected = random.sample(
        images,
        min(IMAGES_PER_CLASS, len(images)),
    )

    for col in range(IMAGES_PER_CLASS):
        ax = axes[row][col]

        if col >= len(selected):
            ax.axis("off")
            continue

        image = cv2.imread(str(selected[col]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax.imshow(image)
        ax.set_title(class_name, fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.show()