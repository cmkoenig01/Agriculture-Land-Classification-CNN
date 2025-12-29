"""Capstone evaluation: Land classification using pretrained CNN–ViT hybrid models (Keras + PyTorch).

Run from repo root:
  python src/evaluate_hybrid_models.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from data_download import prepare_dataset
from model_download import download_capstone_models
from keras_hybrid_loader import load_keras_hybrid_model
from metrics import print_metrics


def ensure_dataset_root(extract_dir: str | Path = ".") -> Path:
    prepare_dataset(extract_dir)
    return Path(extract_dir) / "images_dataSAT"


def eval_keras(model_path: str | Path, dataset_root: str | Path, img_size=(224, 224), batch_size=32):
    import tensorflow as tf

    model = load_keras_hybrid_model(str(model_path))

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(dataset_root),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = ds.class_names

    y_true, y_prob = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_prob.append(probs)
        y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    return class_names, y_true, y_pred, y_prob


def eval_pytorch(state_dict_path: str | Path, dataset_root: str | Path, img_size=224, batch_size=32):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from pytorch_hybrid_model import CNN_ViT_Hybrid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(dataset_root), transform=tfm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = CNN_ViT_Hybrid(num_classes=len(dataset.classes)).to(device)
    sd = torch.load(str(state_dict_path), map_location=device)
    model.load_state_dict(sd)
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            y_true.append(y.numpy())
            y_pred.append(torch.argmax(probs, dim=1).cpu().numpy())
            y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return dataset.classes, y_true, y_pred, y_prob


def main():
    dataset_root = ensure_dataset_root(".")
    model_paths = download_capstone_models("models")

    class_labels, y_true_k, y_pred_k, y_prob_k = eval_keras(model_paths["keras_model"], dataset_root)
    print_metrics(y_true_k, y_pred_k, y_prob_k, class_labels, "Keras CNN–ViT Hybrid")

    try:
        class_labels_pt, y_true_t, y_pred_t, y_prob_t = eval_pytorch(model_paths["pytorch_state_dict"], dataset_root)
        print_metrics(y_true_t, y_pred_t, y_prob_t, class_labels_pt, "PyTorch CNN–ViT Hybrid")
    except Exception as e:
        print("\nPyTorch evaluation skipped (likely state_dict mismatch).")
        print("Error:", e)


if __name__ == "__main__":
    main()
