"""Vision Transformer (ViT) implementation using PyTorch."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange


IMAGE_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBED_DIM = 64
NUM_HEADS = 4
DEPTH = 6
MLP_DIM = 128


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.patch_embed = PatchEmbedding(IMAGE_SIZE, PATCH_SIZE, embed_dim=EMBED_DIM)
        num_patches = self.patch_embed.num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, EMBED_DIM))

        self.transformer = nn.Sequential(*[
            TransformerEncoder(EMBED_DIM, NUM_HEADS, MLP_DIM) for _ in range(DEPTH)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)


def get_dataloader(data_dir: str, image_size: int = IMAGE_SIZE):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=tfm)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return loader, len(dataset.classes)


def train(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f} | Accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    DATA_DIR = "images_dataSAT"
    loader, num_classes = get_dataloader(DATA_DIR)

    model = VisionTransformer(num_classes=num_classes).to(device)
    train(model, loader, device)

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/vit_pytorch_model.pth")
    print("Saved: models/vit_pytorch_model.pth")
