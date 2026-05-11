from __future__ import annotations

from .offset_correction import OffsetCorrectionModel, fit_offset_correction


def fit_residual_correction(*args, **kwargs) -> OffsetCorrectionModel:
    kwargs.setdefault("residual_scale", 1.0)
    return fit_offset_correction(*args, **kwargs)


__all__ = ["OffsetCorrectionModel", "fit_residual_correction"]
