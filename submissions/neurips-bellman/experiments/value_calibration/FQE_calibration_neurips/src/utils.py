from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml


Array = np.ndarray


@dataclass
class TimedBlock:
    seconds: float = 0.0


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def train_calibration_split(n: int, train_fraction: float, seed: int) -> tuple[Array, Array]:
    rng = rng_from_seed(seed)
    indices = rng.permutation(np.arange(n))
    n_train = int(round(float(train_fraction) * n))
    n_train = min(max(n_train, 1), n - 1)
    return np.sort(indices[:n_train]), np.sort(indices[n_train:])


def kfold_indices(n: int, n_folds: int, seed: int) -> list[tuple[Array, Array]]:
    rng = rng_from_seed(seed)
    indices = rng.permutation(np.arange(n))
    folds = np.array_split(indices, int(n_folds))
    out = []
    for fold_id in range(len(folds)):
        valid = np.sort(folds[fold_id])
        train = np.sort(np.concatenate([folds[j] for j in range(len(folds)) if j != fold_id]))
        out.append((train, valid))
    return out


def mse(x: Array, y: Array) -> float:
    return float(np.mean((np.asarray(x) - np.asarray(y)) ** 2))


def finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


@contextmanager
def timed() -> Iterator[TimedBlock]:
    block = TimedBlock()
    start = time.perf_counter()
    try:
        yield block
    finally:
        block.seconds = time.perf_counter() - start


def safe_mean(values: Array) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def result_path(results_dir: str | Path, stem: str, suffix: str = ".csv") -> Path:
    return ensure_dir(results_dir) / f"{stem}{suffix}"
