"""Download + extract the IBM Skills Network image dataset used in the labs.

Dataset URL (used throughout the notebooks):
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple

DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"
)


def download_file(url: str, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    print(f"Downloading: {url}")
    print(f"To: {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def extract_tar(tar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {tar_path} -> {extract_dir}")

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_dir)

    return extract_dir


def prepare_dataset(
    extract_dir: str | Path = ".",
    url: str = DATA_URL,
    tar_name: str = "images-dataSAT.tar",
) -> Tuple[Path, Path]:
    """Download + extract the dataset if needed.

    Returns:
        (dir_non_agri, dir_agri)
    """
    extract_dir = Path(extract_dir)
    tar_path = extract_dir / tar_name

    download_file(url, tar_path)
    extract_tar(tar_path, extract_dir)

    base_dir = extract_dir / "images_dataSAT"
    dir_non_agri = base_dir / "class_0_non_agri"
    dir_agri = base_dir / "class_1_agri"

    if not dir_non_agri.exists() or not dir_agri.exists():
        raise FileNotFoundError(
            "Expected extracted folders not found. "
            f"Looked for:\n  {dir_non_agri}\n  {dir_agri}\n"
            "Check extraction path and tar contents."
        )

    return dir_non_agri, dir_agri


if __name__ == "__main__":
    non_agri_dir, agri_dir = prepare_dataset(".")
    print("Non-agri dir:", non_agri_dir)
    print("Agri dir:", agri_dir)
    print("Non-agri images:", len(list(non_agri_dir.glob("*"))))
    print("Agri images:", len(list(agri_dir.glob("*"))))
