from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows_list for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)
