from __future__ import annotations

from ..data import TransitionBatch
from ..estimators.baselines import fit_estimator
from ..policies import SoftmaxPolicy


def fit_finetuned_estimator(
    learner: str,
    train_batch: TransitionBatch,
    calibration_batch: TransitionBatch,
    n_actions: int,
    policy: SoftmaxPolicy,
    gamma: float,
    params: dict,
    seed: int,
    variant: str = "all_layers",
) -> object:
    """Honest split-sample fine-tuning comparator.

    For neural FQE this function first fits the training split and then runs an
    additional fit on the calibration split initialized from the first stage
    when the implementation supports it. For non-neural learners, the clean
    analogue is a calibration-split refit, which intentionally exposes how much
    power is lost when ordinary retraining is given the same held-out data.
    """

    first = fit_estimator(learner, train_batch, n_actions, policy, gamma, params, seed)
    if learner != "neural_fqe":
        return fit_estimator(learner, calibration_batch, n_actions, policy, gamma, params, seed + 101).model
    from ..estimators.neural_fqe import NeuralFQEConfig, fit_neural_fqe

    cfg_params = dict(params or {})
    cfg_params.setdefault("gamma", gamma)
    cfg = NeuralFQEConfig(**cfg_params)
    if variant == "final_layer":
        cfg = NeuralFQEConfig(**{**cfg.__dict__, "lr": cfg.lr * 0.5, "n_iters": max(2, cfg.n_iters // 3)})
    return fit_neural_fqe(calibration_batch, n_actions, policy, cfg, seed + 101, initial_model=first.model)
