"""Compare memory-based vs generator-based image loading."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Tuple

import numpy as np
from PIL import Image

from data_download import prepare_dataset


@dataclass(frozen=True)
class DatasetPaths:
    non_agri: List[Path]
    agri: List[Path]

    @property
    def all(self) -> List[Path]:
        return self.non_agri + self.agri


def list_image_paths(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file()])


def build_dataset_paths(extract_dir: str | Path = ".") -> DatasetPaths:
    dir_non_agri, dir_agri = prepare_dataset(extract_dir)
    return DatasetPaths(
        non_agri=list_image_paths(dir_non_agri),
        agri=list_image_paths(dir_agri),
    )


def load_image_as_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img)


def memory_based_load(paths: Iterable[Path]) -> np.ndarray:
    images: List[np.ndarray] = []
    for p in paths:
        images.append(load_image_as_array(p))
    return np.array(images, dtype=object)


def generator_based_load(paths: Iterable[Path]) -> Generator[np.ndarray, None, None]:
    for p in paths:
        yield load_image_as_array(p)


def benchmark_memory(paths: List[Path]) -> Tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    arr = memory_based_load(paths)
    t1 = time.perf_counter()
    return (t1 - t0), arr


def benchmark_generator(paths: List[Path]) -> float:
    t0 = time.perf_counter()
    for _img in generator_based_load(paths):
        pass
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    ds = build_dataset_paths(".")
    print(f"Non-agri count: {len(ds.non_agri)}")
    print(f"Agri count:     {len(ds.agri)}")
    print(f"Total:          {len(ds.all)}")

    target = ds.non_agri

    mem_time, mem_arr = benchmark_memory(target)
    gen_time = benchmark_generator(target)

    print("\n--- Benchmark Results ---")
    print(f"Memory-based load time:    {mem_time:.4f} seconds")
    print(f"Generator-based load time: {gen_time:.4f} seconds")

    print("\n--- Sanity Check ---")
    print("Memory-based array length:", len(mem_arr))
    first = mem_arr[0]
    if isinstance(first, np.ndarray):
        print("First image shape:", first.shape)


if __name__ == "__main__":
    main()
