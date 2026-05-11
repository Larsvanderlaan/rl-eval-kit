"""Cross-fitting entry points for GenPQR."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from genpqr.datasets import EpisodeDataset, TransitionDataset, ensure_transition_dataset
from genpqr.exceptions import GenPQRConfigurationError
from genpqr.normalization import DiscreteNormalizationPolicy
from genpqr.types import ActionSpaceSpec, Array, NormalizationPolicy


@dataclass
class GenPQRCrossFitResult:
    """Result returned by :func:`fit_genpqr_crossfit`."""

    fold_results: list[Any]
    out_of_fold_rewards: Array
    fold_indices: list[Array]
    final_result: Any | None
    diagnostics: dict[str, Any]

    def predict_reward(self, states: Array, actions: Array) -> Array:
        """Predict rewards with the final refit model."""

        if self.final_result is None:
            raise GenPQRConfigurationError("predict_reward requires refit_final=True.")
        return self.final_result.predict_reward(states, actions)


def fit_genpqr_crossfit(
    *,
    dataset: TransitionDataset | EpisodeDataset | None = None,
    states: Array | None = None,
    actions: Array | None = None,
    next_states: Array | None = None,
    terminals: Array | None = None,
    gamma: float,
    sample_weight: Array | None = None,
    episode_ids: Array | None = None,
    initial_states: Array | None = None,
    initial_actions: Array | None = None,
    env: Any | None = None,
    action_space: ActionSpaceSpec | None = None,
    normalization_policy: NormalizationPolicy | None = None,
    anchor_function: Callable[[Array], Array] | float = 0.0,
    config: Any | None = None,
    n_folds: int = 5,
    seed: int = 123,
    refit_final: bool = True,
    episode_respecting: bool = True,
) -> GenPQRCrossFitResult:
    """Fit GenPQR with deterministic cross-fitting."""

    from genpqr.api import GenPQRConfig, fit_genpqr

    data = ensure_transition_dataset(
        dataset=dataset,
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        action_space=action_space,
        sample_weight=sample_weight,
        episode_ids=episode_ids,
        initial_states=initial_states,
        initial_actions=initial_actions,
    )
    cfg = GenPQRConfig() if config is None else config
    _validate_crossfit_config(cfg)
    _validate_crossfit_normalization_policy(normalization_policy, data)
    folds = data.make_folds(n_folds=int(n_folds), seed=int(seed), episode_respecting=episode_respecting)
    fold_results: list[Any] = []
    fold_indices: list[Array] = []
    fold_means: list[float] = []
    fold_stds: list[float] = []
    out = np.empty(data.n_rows, dtype=np.float64)
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train = data.subset(train_idx)
        test = data.subset(test_idx)
        fold_cfg = copy.deepcopy(cfg)
        result = fit_genpqr(
            dataset=train,
            gamma=gamma,
            env=env,
            normalization_policy=normalization_policy,
            anchor_function=anchor_function,
            config=fold_cfg,
        )
        rewards = result.predict_reward(test.states, test.actions)
        out[test_idx] = rewards
        fold_means.append(float(np.mean(rewards)))
        fold_stds.append(float(np.std(rewards)))
        result.diagnostics["crossfit_fold"] = int(fold_id)
        result.diagnostics["crossfit_train_rows"] = int(train.n_rows)
        result.diagnostics["crossfit_test_rows"] = int(test.n_rows)
        fold_results.append(result)
        fold_indices.append(np.asarray(test_idx, dtype=np.int64))
    final_result = None
    if refit_final:
        final_result = fit_genpqr(
            dataset=data,
            gamma=gamma,
            env=env,
            normalization_policy=normalization_policy,
            anchor_function=anchor_function,
            config=copy.deepcopy(cfg),
        )
        final_rewards = final_result.predict_reward(data.states, data.actions)
        final_refit_correlation = _safe_corr(out, final_rewards)
        final_refit_mean_delta = float(np.mean(final_rewards) - np.mean(out))
    else:
        final_refit_correlation = None
        final_refit_mean_delta = None
    diagnostics = {
        "n_folds": int(n_folds),
        "n_rows": int(data.n_rows),
        "refit_final": bool(refit_final),
        "out_of_fold_reward_mean": float(np.mean(out)),
        "out_of_fold_reward_std": float(np.std(out)),
        "fold_reward_mean_spread": float(np.max(fold_means) - np.min(fold_means)) if fold_means else None,
        "fold_reward_std_spread": float(np.max(fold_stds) - np.min(fold_stds)) if fold_stds else None,
        "final_refit_reward_correlation": final_refit_correlation,
        "final_refit_reward_mean_delta": final_refit_mean_delta,
        "failed_fold_errors": [],
        "fold_test_sizes": [int(idx.shape[0]) for idx in fold_indices],
    }
    return GenPQRCrossFitResult(
        fold_results=fold_results,
        out_of_fold_rewards=out,
        fold_indices=fold_indices,
        final_result=final_result,
        diagnostics=diagnostics,
    )


def _validate_crossfit_config(config: Any) -> None:
    for field in ("policy", "q"):
        value = getattr(config, field)
        if isinstance(value, str):
            continue
        try:
            copy.deepcopy(value)
        except Exception as exc:  # pragma: no cover - depends on user objects.
            raise GenPQRConfigurationError(
                f"config.{field} must be named or deepcopy-able for cross-fitting."
            ) from exc


def _validate_crossfit_normalization_policy(
    normalization_policy: NormalizationPolicy | None,
    data: TransitionDataset,
) -> None:
    if normalization_policy is None:
        return
    if isinstance(normalization_policy, DiscreteNormalizationPolicy) and not callable(normalization_policy.probabilities):
        probs = np.asarray(normalization_policy.probabilities)
        if probs.ndim == 2 and probs.shape[0] == data.n_rows:
            raise GenPQRConfigurationError(
                "fit_genpqr_crossfit does not support full-data row-bound DiscreteNormalizationPolicy "
                "matrices because fold models must predict held-out rewards with held-out normalization rows. "
                "Use a probability vector or callable normalization policy."
            )


def _safe_corr(left: Array, right: Array) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return None
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])
