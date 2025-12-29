"""Download pretrained capstone models used for CNN–ViT integration/evaluation."""

from __future__ import annotations

from pathlib import Path
import urllib.request


def download_file(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    print(f"Downloading:\n  {url}\n-> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def download_capstone_models(output_dir: str | Path = "models") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keras_model_url = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "7uNMQhNyTA8qSSDGn5Cc7A/keras-cnn-vit-ai-capstone.keras"
    )
    pytorch_state_dict_url = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "rFBrDlu1NNcAzir5Uww8eg/pytorch-cnn-vit-ai-capstone-model-state-dict.pth"
    )

    keras_path = output_dir / "keras_cnn_vit_ai_capstone.keras"
    torch_path = output_dir / "pytorch_cnn_vit_ai_capstone_state_dict.pth"

    download_file(keras_model_url, keras_path)
    download_file(pytorch_state_dict_url, torch_path)

    return {"keras_model": keras_path, "pytorch_state_dict": torch_path}


if __name__ == "__main__":
    print(download_capstone_models("models"))
