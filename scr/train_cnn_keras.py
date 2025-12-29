"""Train and evaluate a CNN classifier using Keras."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    data_dir: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=validation_split)

    train_gen = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model(
    data_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
):
    train_gen, val_gen = create_data_generators(data_dir, image_size=image_size, batch_size=batch_size)

    model = build_cnn_model(input_shape=(*image_size, 3), num_classes=train_gen.num_classes)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    return model, history


if __name__ == "__main__":
    DATA_DIR = "images_dataSAT"

    model, history = train_model(DATA_DIR, epochs=10, batch_size=32)
    plot_training_history(history)

    Path("models").mkdir(exist_ok=True)
    model.save("models/cnn_keras_model.h5")
    print("Saved: models/cnn_keras_model.h5")
