from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


class SerializableEstimatorMixin:
    """Small pickle-based persistence mixin for fitted estimators."""

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Any:
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object has type {type(obj)!r}, expected {cls!r}.")
        return obj
