from __future__ import annotations

from typing import Any

from ..calibration.protocols import ProtocolContext, _evaluate_row
from ..estimators.baselines import fit_estimator
from ..utils import timed, train_calibration_split
from .finetuning import fit_finetuned_estimator
from .offset_correction import fit_offset_correction
from .regularized_toward_first_stage import fit_regularized_toward_first_stage
from .residual_correction import fit_residual_correction


def run_split_comparator(
    ctx: ProtocolContext,
    comparator: str,
    train_fraction: float,
    calibration_target: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    train_idx, cal_idx = train_calibration_split(len(ctx.batch), train_fraction, ctx.seed + 313)
    train_batch = ctx.batch.subset(train_idx)
    cal_batch = ctx.batch.subset(cal_idx)
    with timed() as tb:
        if comparator.startswith("fine_tuning"):
            variant = "final_layer" if comparator.endswith("final_layer") else "all_layers"
            model = fit_finetuned_estimator(
                ctx.learner,
                train_batch,
                cal_batch,
                ctx.env.n_actions,
                ctx.target_policy,
                ctx.gamma,
                ctx.learner_params,
                ctx.seed,
                variant=variant,
            )
        else:
            base = fit_estimator(
                ctx.learner, train_batch, ctx.env.n_actions, ctx.target_policy, ctx.gamma, ctx.learner_params, ctx.seed
            ).model
            if comparator == "offset_correction":
                model = fit_offset_correction(base, cal_batch, ctx.gamma, ctx.env.n_actions, ctx.seed, **params)
            elif comparator == "residual_correction":
                model = fit_residual_correction(base, cal_batch, ctx.gamma, ctx.env.n_actions, ctx.seed, **params)
            elif comparator == "regularized_toward_first_stage":
                model = fit_regularized_toward_first_stage(base, cal_batch, ctx.gamma, ctx.env.n_actions, ctx.seed, **params)
            else:
                raise ValueError(f"Unknown split comparator '{comparator}'.")
    return _evaluate_row(ctx, model, None, {
        "calibrated": False,
        "protocol": comparator,
        "calibrator": "not_calibration",
        "calibration_target": calibration_target,
        "all_data": False,
        "sample_splitting": True,
        "train_fraction": train_fraction,
        "calibration_fraction": 1.0 - train_fraction,
    }, tb.seconds)
