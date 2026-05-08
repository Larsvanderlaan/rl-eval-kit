from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import sparse


Array = np.ndarray


def leaf_assignments_to_csr(
    assignments: Array,
    *,
    n_features: int | None = None,
    offset: int = 0,
    scale: float = 1.0,
) -> sparse.csr_matrix:
    idx = np.asarray(assignments, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        width = int(n_features or 0)
        return sparse.csr_matrix((0, width), dtype=np.float64)
    if np.min(idx) < 0:
        raise ValueError("assignments must be nonnegative contiguous feature ids.")
    width = int(n_features) if n_features is not None else int(np.max(idx) + 1)
    rows = np.arange(idx.size, dtype=np.int64)
    cols = idx + int(offset)
    data = np.full(idx.size, float(scale), dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(idx.size, int(offset) + width))


def remap_labels(labels: Array, known_labels: Sequence[int] | None = None) -> tuple[Array, list[int]]:
    raw = np.asarray(labels, dtype=np.int64).reshape(-1)
    if known_labels is None:
        keys = sorted(int(x) for x in np.unique(raw))
    else:
        keys = [int(x) for x in known_labels]
    mapping = {key: pos for pos, key in enumerate(keys)}
    out = np.empty(raw.shape[0], dtype=np.int64)
    for i, val in enumerate(raw):
        try:
            out[i] = mapping[int(val)]
        except KeyError as exc:
            raise ValueError(f"Unknown leaf label {int(val)}.") from exc
    return out, keys


def average_next_features(
    transform: Callable[[Array], sparse.csr_matrix],
    X_next: Array,
) -> sparse.csr_matrix:
    xp = np.asarray(X_next, dtype=np.float64)
    if xp.ndim == 2:
        return transform(xp).tocsr()
    if xp.ndim != 3:
        raise ValueError(f"X_next must be 2D or 3D, got shape {xp.shape}.")
    n, m, p = xp.shape
    flat = transform(xp.reshape(n * m, p)).tocsr()
    rows = np.repeat(np.arange(n, dtype=np.int64), m)
    cols = np.arange(n * m, dtype=np.int64)
    data = np.full(n * m, 1.0 / float(m), dtype=np.float64)
    aggregator = sparse.csr_matrix((data, (rows, cols)), shape=(n, n * m))
    return (aggregator @ flat).tocsr()


def hstack_csr(blocks: Sequence[sparse.csr_matrix]) -> sparse.csr_matrix:
    if not blocks:
        raise ValueError("at least one feature block is required.")
    return sparse.hstack([block.tocsr() for block in blocks], format="csr", dtype=np.float64)


def leaf_matrix_from_columns(
    leaves: Array,
    *,
    column_maps: list[dict[int, int]] | None = None,
    scale: float | None = None,
) -> tuple[sparse.csr_matrix, list[dict[int, int]]]:
    arr = np.asarray(leaves)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"leaf array must be 1D or 2D after squeezing, got shape {arr.shape}.")
    arr = arr.astype(np.int64, copy=False)
    n, t = arr.shape
    if column_maps is None:
        column_maps = []
        for j in range(t):
            keys = sorted(int(x) for x in np.unique(arr[:, j]))
            column_maps.append({key: pos for pos, key in enumerate(keys)})
    if len(column_maps) != t:
        raise ValueError("column_maps length does not match number of leaf columns.")
    offsets = np.cumsum([0] + [len(m) for m in column_maps[:-1]])
    width = int(sum(len(m) for m in column_maps))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    value = 1.0 / float(t) if scale is None else float(scale)
    for j, mapping in enumerate(column_maps):
        mapped = np.array([mapping.get(int(v), -1) for v in arr[:, j]], dtype=np.int64)
        keep = mapped >= 0
        if not np.any(keep):
            continue
        rows.append(np.nonzero(keep)[0].astype(np.int64))
        cols.append((mapped[keep] + int(offsets[j])).astype(np.int64))
        data.append(np.full(int(np.sum(keep)), value, dtype=np.float64))
    if rows:
        matrix = sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, width),
        )
    else:
        matrix = sparse.csr_matrix((n, width), dtype=np.float64)
    return matrix, column_maps
