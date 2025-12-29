"""Data loading and augmentation using Keras ImageDataGenerator."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    data_dir: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    train_generator = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def visualize_batch(generator, num_images: int = 6):
    images, _labels = next(generator)

    plt.figure(figsize=(12, 6))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")

    plt.suptitle("Augmented Image Samples")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATA_DIR = "images_dataSAT"

    train_gen, val_gen = create_data_generators(DATA_DIR)

    print("Classes:", train_gen.class_indices)
    print("Training samples:", train_gen.samples)
    print("Validation samples:", val_gen.samples)

    visualize_batch(train_gen)
