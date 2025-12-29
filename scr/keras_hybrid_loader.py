"""Custom layers for loading the pretrained Keras CNN–ViT hybrid model."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.drop1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=None):
        attn_out = self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.drop1(attn_out, training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return cfg


def load_keras_hybrid_model(path: str):
    from tensorflow.keras.models import load_model
    return load_model(
        path,
        custom_objects={
            "AddPositionEmbedding": AddPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        },
    )
