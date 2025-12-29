"""PyTorch CNN–ViT Hybrid model (clean implementation).

Note: To load a provided pretrained state_dict perfectly, the module names and
tensor shapes must match exactly. If load_state_dict errors, tweak this file
to match the state_dict keys/shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
        )

    def forward(self, x):
        return self.features(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHead(nn.Module):
    def __init__(self, in_ch: int = 1024, embed_dim: int = 256, depth: int = 4, heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_ch, embed_dim)
        self.blocks = nn.Sequential(*[TransformerEncoderBlock(embed_dim, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, feat_map):
        b, c, h, w = feat_map.shape
        x = feat_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class CNN_ViT_Hybrid(nn.Module):
    def __init__(self, num_classes: int = 2, embed_dim: int = 256, depth: int = 4, heads: int = 4):
        super().__init__()
        self.cnn = ConvNet()
        self.vit = ViTHead(in_ch=1024, embed_dim=embed_dim, depth=depth, heads=heads, num_classes=num_classes)

    def forward(self, x):
        feats = self.cnn(x)
        return self.vit(feats)
