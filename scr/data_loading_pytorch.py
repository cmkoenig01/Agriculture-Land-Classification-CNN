"""Data loading and augmentation using PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_transform, val_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def inspect_batch(dataloader: DataLoader):
    images, labels = next(iter(dataloader))
    print("Image batch shape:", tuple(images.shape))
    print("Label batch shape:", tuple(labels.shape))


if __name__ == "__main__":
    DATA_DIR = "images_dataSAT"

    train_loader, val_loader = create_dataloaders(DATA_DIR)

    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))

    inspect_batch(train_loader)
