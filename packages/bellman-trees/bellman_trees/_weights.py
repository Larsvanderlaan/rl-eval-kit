from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilizedWeights:
    values: np.ndarray
    diagnostics: dict[str, float | None]


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total_sq = float(np.sum(w**2))
    if total_sq <= 0.0:
        return 0.0
    return float(np.sum(w) ** 2 / total_sq)


def stabilize_weights(
    weights: np.ndarray | None,
    n_samples: int | None = None,
    *,
    min_weight: float = 1e-8,
    max_weight: float | None = None,
    clip_quantile: float | None = 0.995,
    uniform_mix: float = 0.0,
    target_ess_fraction: float | None = None,
    max_uniform_mix: float = 0.75,
    normalize: bool = True,
) -> StabilizedWeights:
    """Clip, optionally shrink, and normalize nonnegative importance weights."""

    if weights is None:
        if n_samples is None:
            raise ValueError("n_samples is required when weights is None.")
        raw = np.ones(int(n_samples), dtype=np.float64)
    else:
        raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if n_samples is not None and raw.shape[0] != int(n_samples):
            raise ValueError(f"weights must have length {n_samples}, got {raw.shape[0]}.")
    if raw.size == 0:
        raise ValueError("weights must be nonempty.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("weights contain non-finite values.")

    upper = None
    if clip_quantile is not None:
        if not 0.0 < float(clip_quantile) <= 1.0:
            raise ValueError("clip_quantile must be in (0, 1].")
        upper = float(np.quantile(raw, float(clip_quantile)))
    if max_weight is not None:
        upper = min(float(max_weight), upper) if upper is not None else float(max_weight)
    w = np.maximum(raw, float(min_weight))
    if upper is not None:
        w = np.minimum(w, max(float(min_weight), upper))
    if normalize:
        mean = float(np.mean(w))
        if mean <= 0.0:
            raise ValueError("weights have non-positive mean after clipping.")
        w = w / mean

    ess_before = effective_sample_size(w) / max(w.size, 1)
    chosen_mix = float(max(0.0, uniform_mix))

    def mixed(alpha: float) -> np.ndarray:
        cand = (1.0 - alpha) * w + alpha * np.ones_like(w)
        return cand / np.mean(cand) if normalize else cand

    if target_ess_fraction is not None and ess_before < float(target_ess_fraction):
        lo = chosen_mix
        hi = max(lo, float(max_uniform_mix))
        if effective_sample_size(mixed(hi)) / max(w.size, 1) < float(target_ess_fraction):
            hi = 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if effective_sample_size(mixed(mid)) / max(w.size, 1) >= float(target_ess_fraction):
                hi = mid
            else:
                lo = mid
        chosen_mix = hi

    w = mixed(chosen_mix) if chosen_mix > 0.0 else w
    if normalize:
        w = w / np.mean(w)
    diagnostics = {
        "raw_mean": float(np.mean(raw)),
        "raw_max": float(np.max(raw)),
        "effective_max_weight": None if upper is None else float(upper),
        "chosen_uniform_mix": float(chosen_mix),
        "ess_fraction_before_mix": float(ess_before),
        "ess_fraction_after_mix": float(effective_sample_size(w) / max(w.size, 1)),
        "weight_mean": float(np.mean(w)),
        "weight_max": float(np.max(w)),
    }
    return StabilizedWeights(values=np.ascontiguousarray(w, dtype=np.float64), diagnostics=diagnostics)
