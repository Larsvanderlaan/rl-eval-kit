from __future__ import annotations

import numpy as np


try:  # pragma: no cover - exercised only when the optional extra is installed.
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - default dependency set has no numba.
    HAS_NUMBA = False
    njit = None  # type: ignore[assignment]


if HAS_NUMBA:  # pragma: no cover

    @njit(cache=True)
    def _apply_tree_kernel(
        children_left,
        children_right,
        feature_index,
        threshold_bin,
        default_left,
        leaf_index,
        leaf_value,
        bins,
        missing,
    ):
        n = bins.shape[0]
        leaves = np.empty(n, dtype=np.int32)
        values = np.empty(n, dtype=np.float64)
        for i in range(n):
            node = 0
            while children_left[node] >= 0:
                feature = feature_index[node]
                if missing[i, feature]:
                    if default_left[node]:
                        node = children_left[node]
                    else:
                        node = children_right[node]
                elif bins[i, feature] <= threshold_bin[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            leaves[i] = leaf_index[node]
            values[i] = leaf_value[node]
        return leaves, values


def apply_tree_numba(
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature_index: np.ndarray,
    threshold_bin: np.ndarray,
    default_left: np.ndarray,
    leaf_index: np.ndarray,
    leaf_value: np.ndarray,
    bins: np.ndarray,
    missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not HAS_NUMBA:
        return None
    return _apply_tree_kernel(  # type: ignore[name-defined]
        children_left,
        children_right,
        feature_index,
        threshold_bin,
        default_left,
        leaf_index,
        leaf_value,
        bins,
        missing,
    )
