"""Train a Vision Transformer (ViT) classifier using Keras."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 6
MLP_HEAD_UNITS = [128, 64]
BATCH_SIZE = 32
EPOCHS = 10


def load_data(data_dir: str):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, train_ds.class_names


class Patches(layers.Layer):
    def __init__(self, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg


def create_vit_classifier(num_classes: int):
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    patches = Patches(PATCH_SIZE)(inputs)
    encoded = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        mlp = keras.Sequential([
            layers.Dense(MLP_HEAD_UNITS[0], activation="gelu"),
            layers.Dense(MLP_HEAD_UNITS[1], activation="gelu"),
        ])
        encoded = layers.Add()([mlp(x3), x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    logits = layers.Dense(num_classes)(representation)
    return keras.Model(inputs=inputs, outputs=logits)


def train_vit(data_dir: str):
    train_ds, val_ds, class_names = load_data(data_dir)
    model = create_vit_classifier(num_classes=len(class_names))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    return model, history


if __name__ == "__main__":
    DATA_DIR = "images_dataSAT"
    model, _history = train_vit(DATA_DIR)

    Path("models").mkdir(exist_ok=True)
    model.save("models/vit_keras_model.keras")
    print("Saved: models/vit_keras_model.keras")
