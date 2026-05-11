"""Two-stage FORI model selection by adjoint Bellman error.

This module implements a production-oriented model-selection layer for fitted
occupancy-ratio iteration (FORI). It is intentionally a selector/evaluator:
the primary criterion is a held-out adjoint Bellman error (ABE), not a
minimax/witness estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Sequence
import warnings

import numpy as np

from occupancy_ratio.configs import ActionRatioConfig, OccupancyRegressionConfig, SourceStateRatioConfig
from occupancy_ratio.fit_occupancy_ratio import fit_discounted_occupancy_ratio
from occupancy_ratio._tuning_first_stage import (
    _density_ratio_score,
    _fit_action,
    _fit_generic_density,
    _predict_action_query,
    _predict_generic_density,
)


Array = np.ndarray
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FORITwoStageCVConfig:
    """Configuration for two-stage FORI ABE model selection.

    Parameters
    ----------
    split_fractions:
        Fractions for nuisance CV/trainval, FORI candidate train, backup train,
        backup validation, and score splits.
    terminal_mode:
        ``"auto"``, ``"absorbing_state"``, or ``"live_only_submarkov"``.
    backup_regressor_backend:
        ``"lightgbm"`` by default; ``"ridge"`` is dependency-light and useful
        for small deterministic tests.
    selection_rule:
        Recommendation rule for ``selected_candidate_id``. The default
        ``"one_se"`` returns the simplest candidate within one bootstrap SE of
        the raw minimum ABE score; use ``"min_score"`` to return the raw
        minimizer.
    one_se_method:
        ``"marginal"`` uses the historical one-SE threshold based on the
        minimum candidate's marginal bootstrap SE. ``"paired"`` uses
        trajectory-bootstrap SEs for paired score differences versus the
        minimum-score candidate. Both selections are always reported.
    backup_rank_selection_metric:
        Criterion used to choose the low-rank adjoint backup rank on
        ``D_backup_val``. ``"regression_mse"`` is the historical held-out
        adjoint-regression loss, ``"abe_val"`` uses a validation adjoint
        Bellman residual, and ``"direct_agreement"`` compares low-rank backups
        to the direct multi-output backup when feasible.
    """

    split_fractions: tuple[float, float, float, float, float] = (0.35, 0.25, 0.20, 0.10, 0.10)
    first_stage_cv_folds: int = 3
    first_stage_family: str = "boosted"
    first_stage_density_ratio_loss: str = "lsif"
    first_stage_density_ratio_configs: Optional[Sequence[Any]] = None
    terminal_mode: str = "auto"
    treat_timeouts_as_nonterminal: bool = True
    absorbing_observation: Optional[Array] = None
    absorbing_action: Optional[Array] = None
    on_score_leakage: str = "error"
    low_rank_ranks: tuple[int, ...] = (4, 8, 16, 32, 64)
    low_rank_explained_variance: Optional[float] = None
    max_rank: int = 256
    svd_backend: str = "randomized"
    backup_regressor_backend: str = "lightgbm"
    backup_ridge_alpha: float = 1e-6
    backup_lgbm_params: Mapping[str, Any] = field(default_factory=dict)
    backup_lgbm_num_boost_round: int = 80
    enable_direct_multioutput: bool = True
    direct_multioutput_max_candidates: int = 64
    n_bootstrap: int = 200
    selection_rule: str = "one_se"
    one_se_method: str = "marginal"
    backup_rank_selection_metric: str = "regression_mse"
    lambda_negative: float = 1.0
    lambda_second_moment: float = 0.0
    lambda_mean_one: float = 0.0
    enforce_mean_one_ratio: bool = False
    expected_ratio_mass: float = 1.0
    collapse_diagnostic_ess_fraction: float = 0.95
    collapse_diagnostic_weight_cv: float = 0.05
    collapse_diagnostic_min_action_shift: float = 1e-8
    transition_batch_size: int = 100_000
    candidate_block_size: int = 128
    max_memory_mb: int = 4096
    dtype_predictions: str = "float32"
    dtype_accumulation: str = "float64"
    prediction_memmap_dir: Optional[str] = None
    seed: int = 123
    report_value_estimates: bool = False

    def __post_init__(self) -> None:
        if len(self.split_fractions) != 5:
            raise ValueError("split_fractions must contain five entries.")
        if any(float(x) < 0.0 for x in self.split_fractions) or sum(self.split_fractions) <= 0.0:
            raise ValueError("split_fractions must be nonnegative and have positive total mass.")
        if int(self.first_stage_cv_folds) < 2:
            raise ValueError("first_stage_cv_folds must be >= 2.")
        if str(self.first_stage_family) != "boosted":
            raise ValueError("Only first_stage_family='boosted' is currently supported.")
        if str(self.first_stage_density_ratio_loss) not in {"lsif", "logistic", "ulsif"}:
            raise ValueError("first_stage_density_ratio_loss must be 'lsif', 'ulsif', or 'logistic'.")
        if str(self.terminal_mode) not in {"auto", "absorbing_state", "live_only_submarkov"}:
            raise ValueError("terminal_mode must be 'auto', 'absorbing_state', or 'live_only_submarkov'.")
        if str(self.on_score_leakage) not in {"error", "warn", "allow"}:
            raise ValueError("on_score_leakage must be 'error', 'warn', or 'allow'.")
        if str(self.svd_backend) not in {"randomized", "numpy", "torch"}:
            raise ValueError("svd_backend must be 'randomized', 'numpy', or 'torch'.")
        if str(self.backup_regressor_backend) not in {"lightgbm", "ridge", "torch_mlp"}:
            raise ValueError("backup_regressor_backend must be 'lightgbm', 'ridge', or 'torch_mlp'.")
        if int(self.max_rank) <= 0:
            raise ValueError("max_rank must be positive.")
        if int(self.n_bootstrap) < 0:
            raise ValueError("n_bootstrap must be nonnegative.")
        if str(self.selection_rule) not in {"one_se", "min_score"}:
            raise ValueError("selection_rule must be 'one_se' or 'min_score'.")
        if str(self.one_se_method) not in {"marginal", "paired"}:
            raise ValueError("one_se_method must be 'marginal' or 'paired'.")
        if str(self.backup_rank_selection_metric) not in {"regression_mse", "abe_val", "direct_agreement"}:
            raise ValueError("backup_rank_selection_metric must be 'regression_mse', 'abe_val', or 'direct_agreement'.")
        if not (0.0 <= float(self.collapse_diagnostic_ess_fraction) <= 1.0):
            raise ValueError("collapse_diagnostic_ess_fraction must be in [0, 1].")
        if float(self.collapse_diagnostic_weight_cv) < 0.0:
            raise ValueError("collapse_diagnostic_weight_cv must be nonnegative.")
        if float(self.collapse_diagnostic_min_action_shift) < 0.0:
            raise ValueError("collapse_diagnostic_min_action_shift must be nonnegative.")


@dataclass(frozen=True)
class FORICandidateSpec:
    """Candidate ratio source for FORI model selection.

    A candidate may be a fitted model, a callable, cached predictions, or a
    native boosted/neural FORI config. Native candidates are fit on the FORI
    training split in v1.
    """

    candidate_id: str
    family: str = "external"
    model: Any = None
    predictor: Optional[Callable[[Array, Array], Array]] = None
    cached_predictions: Any = None
    occupancy: Any = None
    action_ratio: Any = None
    source_state_ratio: Any = None
    transition_ratio: Any = None
    initial_ratio_mode: str = "auto"
    one_step_ratio_mode: str = "direct"
    iteration: Optional[int] = None
    hyperparams: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    complexity_order_key: Any = None
    trained_on_episode_ids: Optional[Sequence[Any]] = None
    projection_type: str = ""
    damping_alpha: Optional[float] = None


@dataclass
class FirstStageDensityRatioCVResult:
    """Selected first-stage FORI nuisance models."""

    omega0_model: Any
    cpi_model: Any
    selected_omega0_hyperparams: Mapping[str, Any]
    selected_cpi_hyperparams: Mapping[str, Any]
    omega0_cv_table: list[dict[str, Any]]
    cpi_cv_table: list[dict[str, Any]]
    diagnostics: Mapping[str, Any]
    initial_ratio_mode: str
    cpi_ratio_mode: str = "direct"

    def predict_omega0(self, states: Array, actions: Array) -> Array:
        return np.asarray(self.omega0_model.predict(states, actions), dtype=np.float64).reshape(-1)

    def predict_cpi(self, states: Array, actions: Array) -> Array:
        return np.asarray(self.cpi_model.predict(states, actions), dtype=np.float64).reshape(-1)


@dataclass
class FORIModelSelectionResult:
    """Result for two-stage FORI ABE model selection."""

    rows: list[dict[str, Any]]
    selected_candidate_id: Optional[str]
    selected_one_se_candidate_id: Optional[str]
    first_stage: FirstStageDensityRatioCVResult
    split_indices: Mapping[str, Array]
    split_episode_ids: Mapping[str, Array]
    config: FORITwoStageCVConfig
    warnings: list[str] = field(default_factory=list)
    selected_model: Any = None
    selected_min_score_candidate_id: Optional[str] = None
    selection_rule: str = "one_se"
    selected_one_se_marginal_candidate_id: Optional[str] = None
    selected_one_se_paired_candidate_id: Optional[str] = None
    one_se_method: str = "marginal"

    def candidate_rows(self) -> list[dict[str, Any]]:
        """Return one dictionary row per candidate."""
        return [dict(row) for row in self.rows]

    def to_dataframe(self) -> Any:
        """Return result rows as a pandas DataFrame when pandas is available."""
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame(self.rows)


@dataclass
class _FORIData:
    states: Array
    actions: Array
    next_states: Array
    target_actions: Array
    target_next_actions: Array
    episode_ids: Array
    gamma: float
    rewards: Optional[Array]
    initial_states: Optional[Array]
    initial_actions: Optional[Array]
    initial_weights: Optional[Array]
    initial_episode_ids: Optional[Array]
    continuation: Array
    terminal_mode: str
    warnings: list[str]

    @property
    def n(self) -> int:
        return int(self.states.shape[0])

    def current_x(self, idx: Array) -> Array:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        return np.concatenate([self.states[idx], self.actions[idx]], axis=1)

    def successor_x(self, idx: Array) -> Array:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        return np.concatenate([self.next_states[idx], self.target_next_actions[idx]], axis=1)


class FirstStageDensityRatioCV:
    """Tune initial and direct one-step density-ratio nuisances for FORI."""

    def __init__(self, config: Optional[FORITwoStageCVConfig] = None):
        self.config = config if config is not None else FORITwoStageCVConfig()

    def fit(self, data: _FORIData, train_idx: Array) -> FirstStageDensityRatioCVResult:
        """Fit first-stage nuisance models using only ``train_idx`` episodes."""
        start = time.perf_counter()
        train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        folds = _kfold_indices_by_episode_ids(
            data.episode_ids,
            train_idx,
            int(self.config.first_stage_cv_folds),
            int(self.config.seed) + 101,
        )
        configs = _first_stage_configs(self.config)
        initial_mode = _resolve_initial_mode(data)
        omega0_rows: list[dict[str, Any]] = []
        cpi_rows: list[dict[str, Any]] = []
        best_omega: Optional[tuple[float, Any, list[dict[str, Any]]]] = None
        best_cpi: Optional[tuple[float, Any, list[dict[str, Any]]]] = None

        for cfg_id, cfg in enumerate(configs):
            fold_scores: list[float] = []
            error = ""
            for fold_id, valid_idx in enumerate(folds):
                fold_train = _indices_difference(train_idx, valid_idx)
                try:
                    score = self._score_omega0_config(
                        data=data,
                        config=cfg,
                        initial_mode=initial_mode,
                        train_idx=fold_train,
                        valid_idx=valid_idx,
                        seed=int(self.config.seed) + 10_001 * (cfg_id + 1) + fold_id,
                    )
                    fold_scores.append(score)
                    omega0_rows.append(_cv_row("omega0", cfg_id, fold_id, score, cfg, initial_mode, ""))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    omega0_rows.append(_cv_row("omega0", cfg_id, fold_id, float("inf"), cfg, initial_mode, error))
            mean_score = _finite_mean(fold_scores)
            if not error and np.isfinite(mean_score) and (best_omega is None or mean_score < best_omega[0]):
                best_omega = (float(mean_score), cfg, omega0_rows)

        for cfg_id, cfg in enumerate(configs):
            fold_scores = []
            error = ""
            for fold_id, valid_idx in enumerate(folds):
                fold_train = _indices_difference(train_idx, valid_idx)
                try:
                    score = self._score_cpi_config(
                        data=data,
                        config=cfg,
                        train_idx=fold_train,
                        valid_idx=valid_idx,
                        seed=int(self.config.seed) + 20_003 * (cfg_id + 1) + fold_id,
                    )
                    fold_scores.append(score)
                    cpi_rows.append(_cv_row("cpi", cfg_id, fold_id, score, cfg, "direct", ""))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    cpi_rows.append(_cv_row("cpi", cfg_id, fold_id, float("inf"), cfg, "direct", error))
            mean_score = _finite_mean(fold_scores)
            if not error and np.isfinite(mean_score) and (best_cpi is None or mean_score < best_cpi[0]):
                best_cpi = (float(mean_score), cfg, cpi_rows)

        if best_omega is None:
            raise RuntimeError("No omega0 first-stage candidate completed successfully.")
        if best_cpi is None:
            raise RuntimeError("No c_pi first-stage candidate completed successfully.")

        omega_model = self._fit_omega0_model(
            data=data,
            config=best_omega[1],
            initial_mode=initial_mode,
            train_idx=train_idx,
            seed=int(self.config.seed) + 30_007,
        )
        cpi_model = self._fit_cpi_model(
            data=data,
            config=best_cpi[1],
            train_idx=train_idx,
            seed=int(self.config.seed) + 40_009,
        )
        ref_x = data.current_x(train_idx)
        cpi_train = cpi_model.predict(data.states[train_idx], data.actions[train_idx])
        omega_train = omega_model.predict(data.states[train_idx], data.actions[train_idx])
        diagnostics = {
            "runtime_seconds": float(time.perf_counter() - start),
            "omega0_cv_score": float(best_omega[0]),
            "cpi_cv_score": float(best_cpi[0]),
            "omega0_mean_reference": float(np.mean(omega_train)) if omega_train.size else float("nan"),
            "omega0_second_moment_reference": float(np.mean(omega_train**2)) if omega_train.size else float("nan"),
            "omega0_negative_mass": float(np.mean(np.minimum(omega_train, 0.0) ** 2)) if omega_train.size else float("nan"),
            "cpi_mean_reference": float(np.mean(cpi_train)) if cpi_train.size else float("nan"),
            "cpi_second_moment_reference": float(np.mean(cpi_train**2)) if cpi_train.size else float("nan"),
            "cpi_negative_mass": float(np.mean(np.minimum(cpi_train, 0.0) ** 2)) if cpi_train.size else float("nan"),
            "reference_rows": int(ref_x.shape[0]),
        }
        return FirstStageDensityRatioCVResult(
            omega0_model=omega_model,
            cpi_model=cpi_model,
            selected_omega0_hyperparams=_config_dict(best_omega[1]),
            selected_cpi_hyperparams=_config_dict(best_cpi[1]),
            omega0_cv_table=omega0_rows,
            cpi_cv_table=cpi_rows,
            diagnostics=diagnostics,
            initial_ratio_mode=initial_mode,
        )

    def _score_omega0_config(
        self,
        *,
        data: _FORIData,
        config: Any,
        initial_mode: str,
        train_idx: Array,
        valid_idx: Array,
        seed: int,
    ) -> float:
        if initial_mode == "none":
            return 0.0
        model = self._fit_omega0_model(
            data=data,
            config=config,
            initial_mode=initial_mode,
            train_idx=train_idx,
            seed=seed,
        )
        if isinstance(model, _FactoredInitialRatioModel):
            den_pred = model.source_predict(data.states[valid_idx])
            init_valid = _initial_subset(data, valid_idx)
            if init_valid[0] is None or init_valid[0].shape[0] == 0:
                return float(np.mean(den_pred**2))
            num_pred = model.source_predict(init_valid[0])
            return _density_ratio_score(model.source_fit, den_pred, num_pred, numerator_weights=init_valid[2])
        if isinstance(model, _JointInitialRatioModel):
            den_pred = model.predict(data.states[valid_idx], data.actions[valid_idx])
            init_valid = _initial_subset(data, valid_idx)
            if init_valid[0] is None or init_valid[1] is None or init_valid[0].shape[0] == 0:
                return float(np.mean(den_pred**2))
            num_pred = model.predict(init_valid[0], init_valid[1])
            return _density_ratio_score(model.fit, den_pred, num_pred, numerator_weights=init_valid[2])
        return 0.0

    def _score_cpi_config(self, *, data: _FORIData, config: Any, train_idx: Array, valid_idx: Array, seed: int) -> float:
        model = self._fit_cpi_model(data=data, config=config, train_idx=train_idx, seed=seed)
        pred_ref = model.predict(data.states[valid_idx], data.actions[valid_idx])
        pred_num = model.predict(data.next_states[valid_idx], data.target_next_actions[valid_idx])
        weights = _continuation_weights_for_density(data, valid_idx)
        if weights is not None and str(model.fit.get("density_ratio_loss", "lsif")) != "logistic":
            return _submarkov_density_ratio_score(pred_ref, pred_num, weights)
        return _density_ratio_score(model.fit, pred_ref, pred_num, numerator_weights=weights)

    def _fit_omega0_model(
        self,
        *,
        data: _FORIData,
        config: Any,
        initial_mode: str,
        train_idx: Array,
        seed: int,
    ) -> Any:
        if initial_mode == "none":
            return _ConstantRatioModel(1.0)
        init_train = _initial_subset(data, train_idx)
        if init_train[0] is None or init_train[0].shape[0] == 0:
            return _ConstantRatioModel(1.0)
        family = str(self.config.first_stage_family)
        if initial_mode == "joint":
            if init_train[1] is None:
                raise ValueError("joint initial ratio requires initial_actions.")
            x_ref = data.current_x(train_idx)
            x_num = np.concatenate([init_train[0], init_train[1]], axis=1)
            fit = _fit_generic_density(
                family=family,
                config=config,
                X_ref=x_ref,
                X_num=x_num,
                numerator_weights=init_train[2],
                seed=int(seed),
            )
            return _JointInitialRatioModel(family=family, fit=fit)
        action_fit = _fit_action(
            family=family,
            config=config,
            S=data.states[train_idx],
            A=data.actions[train_idx],
            A_pi=data.target_actions[train_idx],
            seed=int(seed) + 919,
        )
        source_fit = _fit_generic_density(
            family=family,
            config=config,
            X_ref=data.states[train_idx],
            X_num=init_train[0],
            numerator_weights=init_train[2],
            seed=int(seed),
        )
        return _FactoredInitialRatioModel(family=family, source_fit=source_fit, action_fit=action_fit)

    def _fit_cpi_model(self, *, data: _FORIData, config: Any, train_idx: Array, seed: int) -> Any:
        x_ref = data.current_x(train_idx)
        x_num = data.successor_x(train_idx)
        numerator_weights = _continuation_weights_for_density(data, train_idx)
        fit = _fit_generic_density(
            family=str(self.config.first_stage_family),
            config=config,
            X_ref=x_ref,
            X_num=x_num,
            numerator_weights=numerator_weights,
            seed=int(seed),
        )
        # The shared nuisance fitter normalizes numerator weights internally.
        # In live-only sub-Markov mode, c_pi should retain the continuation
        # mass, so predictions are scaled back by E[continuation].
        scale = 1.0 if numerator_weights is None else float(np.mean(np.asarray(numerator_weights, dtype=np.float64)))
        return _DirectDensityRatioModel(family=str(self.config.first_stage_family), fit=fit, scale=scale)


class LowRankAdjointBackupRegressor:
    """Amortized low-rank adjoint backup regressor for candidate ratios."""

    def __init__(
        self,
        *,
        rank: int,
        backend: str = "lightgbm",
        ridge_alpha: float = 1e-6,
        lgbm_params: Optional[Mapping[str, Any]] = None,
        lgbm_num_boost_round: int = 80,
        svd_backend: str = "randomized",
        seed: int = 123,
    ):
        self.rank = int(rank)
        self.backend = str(backend)
        self.ridge_alpha = float(ridge_alpha)
        self.lgbm_params = dict(lgbm_params or {})
        self.lgbm_num_boost_round = int(lgbm_num_boost_round)
        self.svd_backend = str(svd_backend)
        self.seed = int(seed)
        self.w_mean_: Optional[Array] = None
        self.vt_: Optional[Array] = None
        self.singular_values_: Optional[Array] = None
        self.explained_variance_ratio_: Optional[Array] = None
        self.coeff_model_: Any = None

    def fit(self, x_plus: Array, w: Array, *, sample_weight: Optional[Array] = None) -> "LowRankAdjointBackupRegressor":
        x_plus = _as_2d(x_plus, "x_plus")
        w = _as_2d(w, "w").astype(np.float64, copy=False)
        n, m = w.shape
        if x_plus.shape[0] != n:
            raise ValueError("x_plus and w must have aligned rows.")
        weights = _normal_row_weights(sample_weight, n)
        self.w_mean_ = np.sum(weights[:, None] * w, axis=0) / np.sum(weights)
        wc = w - self.w_mean_[None, :]
        rank = min(max(1, int(self.rank)), int(m), int(n))
        vt, singular_values, evr = _fit_truncated_svd(
            wc,
            rank=rank,
            sample_weight=weights,
            backend=self.svd_backend,
            seed=self.seed,
        )
        self.rank = int(vt.shape[0])
        self.vt_ = vt
        self.singular_values_ = singular_values
        self.explained_variance_ratio_ = evr
        z = wc @ vt.T
        self.coeff_model_ = _make_multioutput_regressor(
            backend=self.backend,
            ridge_alpha=self.ridge_alpha,
            lgbm_params=self.lgbm_params,
            lgbm_num_boost_round=self.lgbm_num_boost_round,
            seed=self.seed,
        )
        self.coeff_model_.fit(x_plus, z, sample_weight=weights)
        return self

    def predict(self, x: Array) -> Array:
        if self.w_mean_ is None or self.vt_ is None or self.coeff_model_ is None:
            raise RuntimeError("LowRankAdjointBackupRegressor is not fitted.")
        z_hat = self.coeff_model_.predict(_as_2d(x, "x"))
        return self.w_mean_[None, :] + np.asarray(z_hat, dtype=np.float64) @ self.vt_

    def validation_mse(self, x_plus: Array, w: Array, *, sample_weight: Optional[Array] = None) -> float:
        pred = self.predict(x_plus)
        return _weighted_matrix_mse(w, pred, sample_weight=sample_weight)


class DirectMultiOutputAdjointBackupRegressor:
    """Direct multi-output adjoint backup baseline for small candidate sets."""

    def __init__(
        self,
        *,
        backend: str = "ridge",
        ridge_alpha: float = 1e-6,
        lgbm_params: Optional[Mapping[str, Any]] = None,
        lgbm_num_boost_round: int = 80,
        seed: int = 123,
    ):
        self.backend = str(backend)
        self.ridge_alpha = float(ridge_alpha)
        self.lgbm_params = dict(lgbm_params or {})
        self.lgbm_num_boost_round = int(lgbm_num_boost_round)
        self.seed = int(seed)
        self.model_: Any = None

    def fit(self, x_plus: Array, w: Array, *, sample_weight: Optional[Array] = None) -> "DirectMultiOutputAdjointBackupRegressor":
        w = _as_2d(w, "w")
        self.model_ = _make_multioutput_regressor(
            backend=self.backend,
            ridge_alpha=self.ridge_alpha,
            lgbm_params=self.lgbm_params,
            lgbm_num_boost_round=self.lgbm_num_boost_round,
            seed=self.seed,
        )
        self.model_.fit(_as_2d(x_plus, "x_plus"), w, sample_weight=sample_weight)
        return self

    def predict(self, x: Array) -> Array:
        if self.model_ is None:
            raise RuntimeError("DirectMultiOutputAdjointBackupRegressor is not fitted.")
        return np.asarray(self.model_.predict(_as_2d(x, "x")), dtype=np.float64)


class LowRankAdjointBellmanCV:
    """Low-rank adjoint Bellman CV selector for FORI candidate ratios."""

    def __init__(self, config: Optional[FORITwoStageCVConfig] = None):
        self.config = config if config is not None else FORITwoStageCVConfig()
        self.backup_regressor_: Optional[LowRankAdjointBackupRegressor] = None
        self.rank_used_: Optional[int] = None
        self.adjoint_regression_val_mse_: float = float("nan")
        self.backup_abe_val_score_: float = float("nan")
        self.direct_agreement_val_mse_: float = float("nan")
        self.rank_selection_score_: float = float("nan")
        self.rank_selection_metric_: str = str(self.config.backup_rank_selection_metric)
        self.rank_selection_table_: list[dict[str, Any]] = []

    def fit(
        self,
        *,
        x_plus_train: Array,
        w_train: Array,
        x_plus_val: Array,
        w_val: Array,
        sample_weight_train: Optional[Array] = None,
        sample_weight_val: Optional[Array] = None,
        x_current_val: Optional[Array] = None,
        omega0_val: Optional[Array] = None,
        cpi_val: Optional[Array] = None,
        gamma: Optional[float] = None,
    ) -> "LowRankAdjointBellmanCV":
        m = int(_as_2d(w_train, "w_train").shape[1])
        ranks = _candidate_ranks(self.config, w_train)
        metric = str(self.config.backup_rank_selection_metric)
        if metric == "abe_val" and (
            x_current_val is None or omega0_val is None or cpi_val is None or gamma is None
        ):
            warnings.warn(
                "backup_rank_selection_metric='abe_val' requires validation current-X and nuisance predictions; "
                "falling back to regression_mse.",
                RuntimeWarning,
                stacklevel=2,
            )
            metric = "regression_mse"
        direct_val: Optional[Array] = None
        direct_probe = x_current_val if x_current_val is not None else x_plus_val
        if metric == "direct_agreement":
            if m > int(self.config.direct_multioutput_max_candidates):
                warnings.warn(
                    "backup_rank_selection_metric='direct_agreement' is disabled because the candidate count "
                    "exceeds direct_multioutput_max_candidates; falling back to regression_mse.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                metric = "regression_mse"
            else:
                direct = DirectMultiOutputAdjointBackupRegressor(
                    backend="ridge" if self.config.backup_regressor_backend == "torch_mlp" else self.config.backup_regressor_backend,
                    ridge_alpha=float(self.config.backup_ridge_alpha),
                    lgbm_params=self.config.backup_lgbm_params,
                    lgbm_num_boost_round=int(self.config.backup_lgbm_num_boost_round),
                    seed=int(self.config.seed) + 88_881,
                )
                direct.fit(x_plus_train, w_train, sample_weight=sample_weight_train)
                direct_val = direct.predict(direct_probe)
        best: Optional[tuple[float, LowRankAdjointBackupRegressor, dict[str, float]]] = None
        self.rank_selection_table_ = []
        self.rank_selection_metric_ = metric
        for rank in ranks:
            backup = LowRankAdjointBackupRegressor(
                rank=min(int(rank), m),
                backend=str(self.config.backup_regressor_backend),
                ridge_alpha=float(self.config.backup_ridge_alpha),
                lgbm_params=self.config.backup_lgbm_params,
                lgbm_num_boost_round=int(self.config.backup_lgbm_num_boost_round),
                svd_backend=str(self.config.svd_backend),
                seed=int(self.config.seed) + 7001 + int(rank),
            )
            backup.fit(x_plus_train, w_train, sample_weight=sample_weight_train)
            val_mse = backup.validation_mse(x_plus_val, w_val, sample_weight=sample_weight_val)
            abe_val = float("nan")
            if x_current_val is not None and omega0_val is not None and cpi_val is not None and gamma is not None:
                m_hat_val = backup.predict(x_current_val)
                resid_val = adjoint_bellman_residual(
                    w_score=w_val,
                    omega0_score=omega0_val,
                    cpi_score=cpi_val,
                    m_hat_score=m_hat_val,
                    gamma=float(gamma),
                )
                abe_val = float(np.mean(resid_val**2))
            direct_mse = float("nan")
            if direct_val is not None:
                low_rank_val = backup.predict(direct_probe)
                direct_mse = _weighted_matrix_mse(direct_val, low_rank_val, sample_weight=sample_weight_val)
            if metric == "abe_val":
                selection_score = abe_val
            elif metric == "direct_agreement":
                selection_score = direct_mse
            else:
                selection_score = float(val_mse)
            if not np.isfinite(selection_score):
                selection_score = float("inf")
            row = {
                "rank": float(backup.rank),
                "adjoint_regression_val_mse": float(val_mse),
                "backup_abe_val_score": float(abe_val),
                "direct_agreement_val_mse": float(direct_mse),
                "rank_selection_score": float(selection_score),
                "explained_variance": float(np.sum(backup.explained_variance_ratio_))
                if backup.explained_variance_ratio_ is not None
                else float("nan"),
            }
            self.rank_selection_table_.append(row)
            if best is None or selection_score < best[0]:
                best = (float(selection_score), backup, row)
        if best is None:
            raise RuntimeError("No low-rank adjoint backup rank completed successfully.")
        self.rank_selection_score_, self.backup_regressor_, selected_row = best
        self.rank_used_ = int(self.backup_regressor_.rank)
        self.adjoint_regression_val_mse_ = float(selected_row["adjoint_regression_val_mse"])
        self.backup_abe_val_score_ = float(selected_row["backup_abe_val_score"])
        self.direct_agreement_val_mse_ = float(selected_row["direct_agreement_val_mse"])
        return self

    def score(
        self,
        *,
        x_current_score: Array,
        w_score: Array,
        omega0_score: Array,
        cpi_score: Array,
        gamma: float,
    ) -> dict[str, Array]:
        if self.backup_regressor_ is None:
            raise RuntimeError("LowRankAdjointBellmanCV is not fitted.")
        m_hat = self.backup_regressor_.predict(x_current_score)
        residual = adjoint_bellman_residual(
            w_score=w_score,
            omega0_score=omega0_score,
            cpi_score=cpi_score,
            m_hat_score=m_hat,
            gamma=gamma,
        )
        abe = np.mean(residual**2, axis=0)
        return {"m_hat": m_hat, "residual": residual, "ABE_score": abe}


class FORITwoStageCV:
    """Run two-stage FORI model selection with held-out ABE scoring."""

    def __init__(self, config: Optional[FORITwoStageCVConfig] = None):
        self.config = config if config is not None else FORITwoStageCVConfig()

    def fit(
        self,
        *,
        states: Array,
        actions: Array,
        next_states: Array,
        gamma: float,
        episode_ids: Array,
        target_actions: Optional[Array] = None,
        target_next_actions: Optional[Array] = None,
        target_policy: Any = None,
        action_space: Any = None,
        n_action_samples: int = 1,
        rewards: Optional[Array] = None,
        initial_states: Optional[Array] = None,
        initial_actions: Optional[Array] = None,
        initial_weights: Optional[Array] = None,
        initial_episode_ids: Optional[Array] = None,
        terminated: Optional[Array] = None,
        truncated: Optional[Array] = None,
        done: Optional[Array] = None,
        candidates: Optional[Sequence[FORICandidateSpec]] = None,
    ) -> FORIModelSelectionResult:
        start = time.perf_counter()
        data = _build_data_adapter(
            states=states,
            actions=actions,
            next_states=next_states,
            target_actions=target_actions,
            target_next_actions=target_next_actions,
            target_policy=target_policy,
            action_space=action_space,
            n_action_samples=n_action_samples,
            gamma=gamma,
            episode_ids=episode_ids,
            rewards=rewards,
            initial_states=initial_states,
            initial_actions=initial_actions,
            initial_weights=initial_weights,
            initial_episode_ids=initial_episode_ids,
            terminated=terminated,
            truncated=truncated,
            done=done,
            config=self.config,
        )
        split_indices, split_episode_ids = split_by_episode_ids(data, self.config.split_fractions, self.config.seed)
        _assert_split_disjoint(split_episode_ids)
        score_eps = split_episode_ids["score"]
        candidate_specs = list(candidates or _default_candidate_specs())
        self._check_candidate_leakage(candidate_specs, score_eps)
        first_stage = FirstStageDensityRatioCV(self.config).fit(data, split_indices["nuisance"])
        candidate_specs = self._fit_native_candidates(data, split_indices["fori"], candidate_specs)

        w_backup_train = compute_candidate_ratio_matrix(
            candidate_specs,
            data,
            split_indices["backup_train"],
            split_name="backup_train",
            candidate_block_size=self.config.candidate_block_size,
            transition_batch_size=self.config.transition_batch_size,
            output_memmap_path=_maybe_memmap_path(self.config, "W_backup_train"),
            dtype=self.config.dtype_predictions,
        )
        w_backup_val = compute_candidate_ratio_matrix(
            candidate_specs,
            data,
            split_indices["backup_val"],
            split_name="backup_val",
            candidate_block_size=self.config.candidate_block_size,
            transition_batch_size=self.config.transition_batch_size,
            output_memmap_path=_maybe_memmap_path(self.config, "W_backup_val"),
            dtype=self.config.dtype_predictions,
        )
        w_score = compute_candidate_ratio_matrix(
            candidate_specs,
            data,
            split_indices["score"],
            split_name="score",
            candidate_block_size=self.config.candidate_block_size,
            transition_batch_size=self.config.transition_batch_size,
            output_memmap_path=_maybe_memmap_path(self.config, "W_score"),
            dtype=self.config.dtype_predictions,
        )

        backup_cv = LowRankAdjointBellmanCV(self.config).fit(
            x_plus_train=data.successor_x(split_indices["backup_train"]),
            w_train=np.asarray(w_backup_train, dtype=np.float64),
            x_plus_val=data.successor_x(split_indices["backup_val"]),
            w_val=np.asarray(w_backup_val, dtype=np.float64),
            sample_weight_train=_backup_sample_weight(data, split_indices["backup_train"]),
            sample_weight_val=_backup_sample_weight(data, split_indices["backup_val"]),
            x_current_val=data.current_x(split_indices["backup_val"]),
            omega0_val=first_stage.predict_omega0(data.states[split_indices["backup_val"]], data.actions[split_indices["backup_val"]]),
            cpi_val=first_stage.predict_cpi(data.states[split_indices["backup_val"]], data.actions[split_indices["backup_val"]]),
            gamma=float(data.gamma),
        )
        score_idx = split_indices["score"]
        omega0_score = first_stage.predict_omega0(data.states[score_idx], data.actions[score_idx])
        cpi_score = first_stage.predict_cpi(data.states[score_idx], data.actions[score_idx])
        score_payload = backup_cv.score(
            x_current_score=data.current_x(score_idx),
            w_score=np.asarray(w_score, dtype=np.float64),
            omega0_score=omega0_score,
            cpi_score=cpi_score,
            gamma=float(data.gamma),
        )
        residual = np.asarray(score_payload["residual"], dtype=np.float64)
        abe = np.asarray(score_payload["ABE_score"], dtype=np.float64)
        diagnostics = _candidate_diagnostics(
            np.asarray(w_score, dtype=np.float64),
            action_shift=_policy_action_shift(data, score_idx),
            config=self.config,
        )
        final_score = _final_scores(abe, diagnostics, self.config)
        abe_se, final_se, paired_diff_se, diagnostic_se = _bootstrap_score_ses(
            residual=residual,
            w_score=np.asarray(w_score, dtype=np.float64),
            episode_ids=data.episode_ids[score_idx],
            final_score_components=dict(abe=abe, diagnostics=diagnostics),
            config=self.config,
            selected_idx=int(np.nanargmin(final_score)) if final_score.size else -1,
        )
        min_score_idx = int(np.nanargmin(final_score)) if final_score.size else -1
        one_se_marginal_idx = _one_se_selection(
            final_score,
            final_se,
            candidate_specs,
            min_score_idx,
            method="marginal",
        )
        one_se_paired_idx = _one_se_selection(
            final_score,
            final_se,
            candidate_specs,
            min_score_idx,
            method="paired",
            paired_diff_se=paired_diff_se,
        )
        one_se_idx = one_se_paired_idx if str(self.config.one_se_method) == "paired" else one_se_marginal_idx
        selected_idx = one_se_idx if str(self.config.selection_rule) == "one_se" else min_score_idx
        direct_abe = self._direct_baseline(
            data=data,
            score_idx=score_idx,
            w_train=np.asarray(w_backup_train, dtype=np.float64),
            w_score=np.asarray(w_score, dtype=np.float64),
            omega0_score=omega0_score,
            cpi_score=cpi_score,
            train_idx=split_indices["backup_train"],
            candidates=candidate_specs,
        )
        naive_abe = self._naive_internal_baseline(
            candidates=candidate_specs,
            data=data,
            score_idx=score_idx,
            w_score=np.asarray(w_score, dtype=np.float64),
            omega0_score=omega0_score,
            cpi_score=cpi_score,
        )
        rows = []
        runtime = float(time.perf_counter() - start)
        for j, spec in enumerate(candidate_specs):
            row = _candidate_result_row(
                spec=spec,
                index=j,
                abe=abe,
                abe_se=abe_se,
                final_score=final_score,
                final_score_se=final_se,
                paired_diff_se=paired_diff_se,
                diagnostics=diagnostics,
                diagnostic_se=diagnostic_se,
                direct_abe=direct_abe,
                naive_abe=naive_abe,
                w_score_matrix=np.asarray(w_score, dtype=np.float64),
                selected_idx=min_score_idx,
                one_se_idx=one_se_idx,
                one_se_marginal_idx=one_se_marginal_idx,
                one_se_paired_idx=one_se_paired_idx,
                rank_used=int(backup_cv.rank_used_ or 0),
                adjoint_regression_val_mse=float(backup_cv.adjoint_regression_val_mse_),
                backup_abe_val_score=float(backup_cv.backup_abe_val_score_),
                direct_agreement_val_mse=float(backup_cv.direct_agreement_val_mse_),
                rank_selection_score=float(backup_cv.rank_selection_score_),
                rank_selection_metric=str(backup_cv.rank_selection_metric_),
                rank_selection_table=backup_cv.rank_selection_table_,
                data=data,
                score_idx=score_idx,
                runtime_seconds=runtime,
                config=self.config,
            )
            rows.append(row)
        selected_model = None
        if 0 <= selected_idx < len(candidate_specs):
            selected_model = candidate_specs[selected_idx].model
        LOGGER.info(
            "FORI ABE selected recommended=%s rule=%s one_se_method=%s min=%s one_se=%s paired_one_se=%s rank=%s ABE=%.6g",
            None if selected_idx < 0 else candidate_specs[selected_idx].candidate_id,
            self.config.selection_rule,
            self.config.one_se_method,
            None if min_score_idx < 0 else candidate_specs[min_score_idx].candidate_id,
            None if one_se_idx < 0 else candidate_specs[one_se_idx].candidate_id,
            None if one_se_paired_idx < 0 else candidate_specs[one_se_paired_idx].candidate_id,
            backup_cv.rank_used_,
            float(abe[selected_idx]) if selected_idx >= 0 else float("nan"),
        )
        return FORIModelSelectionResult(
            rows=rows,
            selected_candidate_id=None if selected_idx < 0 else candidate_specs[selected_idx].candidate_id,
            selected_one_se_candidate_id=None if one_se_idx < 0 else candidate_specs[one_se_idx].candidate_id,
            first_stage=first_stage,
            split_indices=split_indices,
            split_episode_ids=split_episode_ids,
            config=self.config,
            selected_min_score_candidate_id=None if min_score_idx < 0 else candidate_specs[min_score_idx].candidate_id,
            selection_rule=str(self.config.selection_rule),
            selected_one_se_marginal_candidate_id=None
            if one_se_marginal_idx < 0
            else candidate_specs[one_se_marginal_idx].candidate_id,
            selected_one_se_paired_candidate_id=None
            if one_se_paired_idx < 0
            else candidate_specs[one_se_paired_idx].candidate_id,
            one_se_method=str(self.config.one_se_method),
            warnings=data.warnings,
            selected_model=selected_model,
        )

    def _fit_native_candidates(
        self,
        data: _FORIData,
        train_idx: Array,
        candidates: Sequence[FORICandidateSpec],
    ) -> list[FORICandidateSpec]:
        out: list[FORICandidateSpec] = []
        for pos, spec in enumerate(candidates):
            if spec.model is not None or spec.predictor is not None or spec.cached_predictions is not None:
                out.append(spec)
                continue
            family = str(spec.family)
            if family not in {"boosted", "neural"}:
                raise ValueError(
                    f"Candidate {spec.candidate_id!r} has no model/predictor/cache and unsupported family {family!r}."
                )
            fit_seed = int(self.config.seed) + 90_001 + 1009 * pos
            if family == "boosted":
                model = fit_discounted_occupancy_ratio(
                    states=data.states[train_idx],
                    actions=data.actions[train_idx],
                    next_states=data.next_states[train_idx],
                    target_actions=data.target_actions[train_idx],
                    gamma=float(data.gamma),
                    initial_states=_initial_subset(data, train_idx)[0],
                    initial_actions=_initial_subset(data, train_idx)[1],
                    initial_weights=_initial_subset(data, train_idx)[2],
                    target_next_actions=data.target_next_actions[train_idx],
                    terminals=1.0 - data.continuation[train_idx],
                    absorbing_state=data.terminal_mode == "absorbing_state",
                    initial_ratio_mode=str(spec.initial_ratio_mode),
                    one_step_ratio_mode=str(spec.one_step_ratio_mode or "direct"),
                    occupancy=replace(
                        spec.occupancy if spec.occupancy is not None else OccupancyRegressionConfig(),
                        seed=fit_seed,
                        show_progress=False,
                    ),
                    action_ratio=replace(
                        spec.action_ratio if spec.action_ratio is not None else ActionRatioConfig(),
                        show_progress=False,
                    ),
                    source_state_ratio=replace(
                        spec.source_state_ratio if spec.source_state_ratio is not None else SourceStateRatioConfig(),
                        show_progress=False,
                    ),
                    transition_ratio=spec.transition_ratio,
                )
            else:
                from occupancy_ratio.fit_occupancy_ratio_neural import (
                    NeuralActionRatioConfig,
                    NeuralOccupancyRegressionConfig,
                    NeuralSourceStateRatioConfig,
                    NeuralTransitionRatioConfig,
                    fit_discounted_occupancy_ratio_neural,
                )

                model = fit_discounted_occupancy_ratio_neural(
                    states=data.states[train_idx],
                    actions=data.actions[train_idx],
                    next_states=data.next_states[train_idx],
                    target_actions=data.target_actions[train_idx],
                    gamma=float(data.gamma),
                    initial_states=_initial_subset(data, train_idx)[0],
                    initial_actions=_initial_subset(data, train_idx)[1],
                    initial_weights=_initial_subset(data, train_idx)[2],
                    target_next_actions=data.target_next_actions[train_idx],
                    terminals=1.0 - data.continuation[train_idx],
                    absorbing_state=data.terminal_mode == "absorbing_state",
                    initial_ratio_mode=str(spec.initial_ratio_mode),
                    one_step_ratio_mode=str(spec.one_step_ratio_mode or "direct"),
                    occupancy=replace(
                        spec.occupancy if spec.occupancy is not None else NeuralOccupancyRegressionConfig(),
                        seed=fit_seed,
                    ),
                    action_ratio=replace(
                        spec.action_ratio if spec.action_ratio is not None else NeuralActionRatioConfig(),
                        seed=fit_seed + 101,
                    ),
                    source_state_ratio=replace(
                        spec.source_state_ratio if spec.source_state_ratio is not None else NeuralSourceStateRatioConfig(),
                        seed=fit_seed + 211,
                    ),
                    transition_ratio=replace(
                        spec.transition_ratio if spec.transition_ratio is not None else NeuralTransitionRatioConfig(),
                        seed=fit_seed + 307,
                    ),
                )
            out.append(replace(spec, model=model, trained_on_episode_ids=np.unique(data.episode_ids[train_idx])))
        return out

    def _check_candidate_leakage(self, candidates: Sequence[FORICandidateSpec], score_eps: Array) -> None:
        score_set = set(np.asarray(score_eps).reshape(-1).tolist())
        for spec in candidates:
            trained = spec.trained_on_episode_ids
            if trained is None and isinstance(spec.metadata, Mapping):
                trained = spec.metadata.get("trained_on_episode_ids")
            if trained is None:
                continue
            overlap = score_set.intersection(np.asarray(trained).reshape(-1).tolist())
            if not overlap:
                continue
            message = (
                f"Candidate {spec.candidate_id!r} metadata overlaps D_score episodes; "
                f"{len(overlap)} score episode(s) would leak into model selection."
            )
            if self.config.on_score_leakage == "error":
                raise ValueError(message)
            if self.config.on_score_leakage == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _direct_baseline(
        self,
        *,
        data: _FORIData,
        score_idx: Array,
        w_train: Array,
        w_score: Array,
        omega0_score: Array,
        cpi_score: Array,
        train_idx: Array,
        candidates: Sequence[FORICandidateSpec],
    ) -> Optional[Array]:
        if not bool(self.config.enable_direct_multioutput):
            return None
        if len(candidates) > int(self.config.direct_multioutput_max_candidates):
            return None
        direct = DirectMultiOutputAdjointBackupRegressor(
            backend="ridge" if self.config.backup_regressor_backend == "torch_mlp" else self.config.backup_regressor_backend,
            ridge_alpha=float(self.config.backup_ridge_alpha),
            lgbm_params=self.config.backup_lgbm_params,
            lgbm_num_boost_round=int(self.config.backup_lgbm_num_boost_round),
            seed=int(self.config.seed) + 99_991,
        )
        direct.fit(data.successor_x(train_idx), w_train, sample_weight=_backup_sample_weight(data, train_idx))
        m_hat = direct.predict(data.current_x(score_idx))
        resid = adjoint_bellman_residual(
            w_score=w_score,
            omega0_score=omega0_score,
            cpi_score=cpi_score,
            m_hat_score=m_hat,
            gamma=float(data.gamma),
        )
        return np.mean(resid**2, axis=0)

    def _naive_internal_baseline(
        self,
        *,
        candidates: Sequence[FORICandidateSpec],
        data: _FORIData,
        score_idx: Array,
        w_score: Array,
        omega0_score: Array,
        cpi_score: Array,
    ) -> Array:
        out = np.full(len(candidates), float("nan"), dtype=np.float64)
        x_current = data.current_x(score_idx)
        for j, spec in enumerate(candidates):
            internal = None
            if isinstance(spec.metadata, Mapping):
                internal = spec.metadata.get("internal_backup_model") or spec.metadata.get("m_internal")
            if internal is None:
                continue
            warnings.warn(
                "naive_internal_ABE can be tautological when computed from the candidate's own backup model; "
                "it is reported only as a diagnostic, not as the primary selector.",
                RuntimeWarning,
                stacklevel=2,
            )
            m_hat = _predict_internal_backup(internal, x_current, data.states[score_idx], data.actions[score_idx])
            resid = adjoint_bellman_residual(
                w_score=w_score[:, [j]],
                omega0_score=omega0_score,
                cpi_score=cpi_score,
                m_hat_score=np.asarray(m_hat, dtype=np.float64).reshape(-1, 1),
                gamma=float(data.gamma),
            )
            out[j] = float(np.mean(resid[:, 0] ** 2))
        return out


def compute_candidate_ratio_matrix(
    candidates: Sequence[FORICandidateSpec],
    data: _FORIData,
    dataset_split: Array,
    *,
    split_name: str,
    candidate_block_size: int = 128,
    transition_batch_size: int = 100_000,
    output_memmap_path: Optional[str | Path] = None,
    dtype: str = "float32",
) -> Array:
    """Compute candidate ratio predictions with candidate/transition blocking."""
    idx = np.asarray(dataset_split, dtype=np.int64).reshape(-1)
    n = int(idx.shape[0])
    m = int(len(candidates))
    if output_memmap_path is None:
        out: Array = np.empty((n, m), dtype=np.dtype(dtype))
    else:
        path = Path(output_memmap_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.memmap(path, mode="w+", dtype=np.dtype(dtype), shape=(n, m))
    block = max(1, int(candidate_block_size))
    batch = max(1, int(transition_batch_size))
    for j0 in range(0, m, block):
        for j, spec in enumerate(candidates[j0 : j0 + block], start=j0):
            cached = _cached_prediction_for_split(spec, split_name, n)
            if cached is not None:
                out[:, j] = cached.astype(out.dtype, copy=False)
                continue
            for i0 in range(0, n, batch):
                rows = idx[i0 : i0 + batch]
                pred = _predict_candidate(spec, data.states[rows], data.actions[rows])
                out[i0 : i0 + rows.shape[0], j] = np.asarray(pred, dtype=out.dtype).reshape(-1)
    if isinstance(out, np.memmap):
        out.flush()
    return out


def adjoint_bellman_residual(
    *,
    w_score: Array,
    omega0_score: Array,
    cpi_score: Array,
    m_hat_score: Array,
    gamma: float,
) -> Array:
    """Compute pointwise FORI adjoint Bellman residuals."""
    w = _as_2d(w_score, "w_score").astype(np.float64, copy=False)
    omega0 = np.asarray(omega0_score, dtype=np.float64).reshape(-1)
    cpi = np.asarray(cpi_score, dtype=np.float64).reshape(-1)
    m_hat = _as_2d(m_hat_score, "m_hat_score").astype(np.float64, copy=False)
    if not (w.shape == m_hat.shape and omega0.shape[0] == w.shape[0] and cpi.shape[0] == w.shape[0]):
        raise ValueError("w_score, m_hat_score, omega0_score, and cpi_score have incompatible shapes.")
    backup = (1.0 - float(gamma)) * omega0[:, None] + float(gamma) * cpi[:, None] * m_hat
    return w - backup


def split_by_episode_ids(
    dataset_or_episode_ids: _FORIData | Array,
    fractions: Sequence[float],
    seed: int,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Split rows by episode into nuisance, FORI, backup, validation, and score sets."""
    if isinstance(dataset_or_episode_ids, _FORIData):
        episode_ids = dataset_or_episode_ids.episode_ids
    else:
        episode_ids = np.asarray(dataset_or_episode_ids)
    groups = np.asarray(episode_ids).reshape(-1)
    unique = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(unique.shape[0])]
    fr = np.asarray(fractions, dtype=np.float64)
    fr = fr / np.sum(fr)
    raw_counts = fr * shuffled.shape[0]
    counts = np.floor(raw_counts).astype(int)
    remainder = int(shuffled.shape[0] - np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        counts[order[:remainder]] += 1
    names = ["nuisance", "fori", "backup_train", "backup_val", "score"]
    split_indices: dict[str, Array] = {}
    split_episodes: dict[str, Array] = {}
    start = 0
    for name, count in zip(names, counts):
        eps = shuffled[start : start + int(count)]
        start += int(count)
        split_episodes[name] = eps
        split_indices[name] = np.flatnonzero(np.isin(groups, eps)).astype(np.int64, copy=False)
    return split_indices, split_episodes


def kfold_by_episode_ids(episode_ids: Array, k: int, seed: int) -> list[Array]:
    """Return global row-index folds that keep episodes intact."""
    all_idx = np.arange(np.asarray(episode_ids).reshape(-1).shape[0], dtype=np.int64)
    return _kfold_indices_by_episode_ids(episode_ids, all_idx, int(k), int(seed))


def sample_target_successor_actions(
    next_obs_batch: Array,
    target_policy: Any,
    action_space: Any = None,
    n_action_samples: int = 1,
    seed: Optional[int] = None,
    *,
    exact_enumeration: bool = False,
) -> Array | dict[str, Array]:
    """Sample or enumerate target-policy successor actions.

    The function supports simple callable policies, policies exposing
    ``sample_action(obs, rng)``, deterministic ``predict(obs)``, and discrete
    enumeration via ``action_probabilities``/``predict_proba`` when available.
    """
    obs = _as_2d(next_obs_batch, "next_obs_batch")
    rng = np.random.default_rng(seed)
    if target_policy is None:
        raise ValueError("target_policy is required when target_next_actions are not supplied.")
    if exact_enumeration:
        n_actions = _discrete_action_count(action_space)
        if n_actions is None:
            raise ValueError("exact_enumeration requires a discrete action_space with n actions.")
        probs = _target_policy_probabilities(target_policy, obs)
        if probs is None:
            raise ValueError("exact_enumeration requires target policy probabilities.")
        actions = np.tile(np.arange(n_actions, dtype=np.int64), obs.shape[0]).reshape(-1, 1)
        row_index = np.repeat(np.arange(obs.shape[0], dtype=np.int64), n_actions)
        return {
            "actions": actions,
            "probabilities": np.asarray(probs, dtype=np.float64).reshape(-1),
            "row_index": row_index,
        }
    samples = []
    reps = max(1, int(n_action_samples))
    for _ in range(reps):
        samples.append(_sample_policy_once(target_policy, obs, rng))
    out = np.vstack(samples) if reps > 1 else samples[0]
    return _as_2d(out, "sampled_actions")


def load_fori_two_stage_config(path: str | Path) -> Mapping[str, Any]:
    """Load a JSON or optional YAML FORI two-stage model-selection config."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError("PyYAML is required to read YAML configs.") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("FORI two-stage config must contain an object.")
    return payload


class _ConstantRatioModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, states: Array, actions: Optional[Array] = None) -> Array:
        del actions
        return np.full(_as_2d(states, "states").shape[0], self.value, dtype=np.float64)


class _DirectDensityRatioModel:
    def __init__(self, *, family: str, fit: Mapping[str, Any], scale: float = 1.0):
        self.family = str(family)
        self.fit = dict(fit)
        self.scale = float(scale)

    def predict(self, states: Array, actions: Array) -> Array:
        x = np.concatenate([_as_2d(states, "states"), _as_2d(actions, "actions")], axis=1)
        return self.scale * _predict_generic_density(self.family, self.fit, x)


class _JointInitialRatioModel(_DirectDensityRatioModel):
    pass


class _FactoredInitialRatioModel:
    def __init__(self, *, family: str, source_fit: Mapping[str, Any], action_fit: Mapping[str, Any]):
        self.family = str(family)
        self.source_fit = dict(source_fit)
        self.action_fit = dict(action_fit)

    def source_predict(self, states: Array) -> Array:
        return _predict_generic_density(self.family, self.source_fit, _as_2d(states, "states"))

    def predict(self, states: Array, actions: Array) -> Array:
        s = _as_2d(states, "states")
        a = _as_2d(actions, "actions")
        source = self.source_predict(s)
        action = _predict_action_query(self.family, self.action_fit, np.concatenate([s, a], axis=1))
        return np.maximum(source * action, 0.0)


class _RidgeMultiOutputRegressor:
    def __init__(self, alpha: float = 1e-6):
        self.alpha = float(alpha)
        self.coef_: Optional[Array] = None

    def fit(self, x: Array, y: Array, *, sample_weight: Optional[Array] = None) -> "_RidgeMultiOutputRegressor":
        x = _as_2d(x, "x").astype(np.float64, copy=False)
        y = _as_2d(y, "y").astype(np.float64, copy=False)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have aligned rows.")
        design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
        w = _normal_row_weights(sample_weight, x.shape[0])
        wx = design * np.sqrt(w)[:, None]
        wy = y * np.sqrt(w)[:, None]
        gram = wx.T @ wx
        penalty = float(self.alpha) * np.eye(gram.shape[0], dtype=np.float64)
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(gram + penalty, wx.T @ wy)
        return self

    def predict(self, x: Array) -> Array:
        if self.coef_ is None:
            raise RuntimeError("Ridge regressor is not fitted.")
        x = _as_2d(x, "x").astype(np.float64, copy=False)
        design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
        return design @ self.coef_


class _LightGBMMultiOutputRegressor:
    def __init__(self, *, params: Mapping[str, Any], num_boost_round: int, seed: int):
        self.params = dict(params)
        self.num_boost_round = int(num_boost_round)
        self.seed = int(seed)
        self.models_: list[Any] = []

    def fit(self, x: Array, y: Array, *, sample_weight: Optional[Array] = None) -> "_LightGBMMultiOutputRegressor":
        import lightgbm as lgb

        x = _as_2d(x, "x").astype(np.float32, copy=False)
        y = _as_2d(y, "y").astype(np.float64, copy=False)
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": max(2, min(50, x.shape[0] // 10 if x.shape[0] else 2)),
            "verbose": -1,
            "seed": self.seed,
        }
        params.update(self.params)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        self.models_ = []
        for j in range(y.shape[1]):
            local_params = dict(params)
            local_params["seed"] = self.seed + 101 * j
            dtrain = lgb.Dataset(x, label=y[:, j], weight=weight, free_raw_data=False)
            self.models_.append(lgb.train(local_params, dtrain, num_boost_round=max(1, int(self.num_boost_round))))
        return self

    def predict(self, x: Array) -> Array:
        if not self.models_:
            raise RuntimeError("LightGBM regressor is not fitted.")
        x = _as_2d(x, "x").astype(np.float32, copy=False)
        return np.column_stack([model.predict(x) for model in self.models_]).astype(np.float64, copy=False)


def _make_multioutput_regressor(
    *,
    backend: str,
    ridge_alpha: float,
    lgbm_params: Mapping[str, Any],
    lgbm_num_boost_round: int,
    seed: int,
) -> Any:
    if backend == "ridge":
        return _RidgeMultiOutputRegressor(alpha=ridge_alpha)
    if backend == "lightgbm":
        return _LightGBMMultiOutputRegressor(params=lgbm_params, num_boost_round=lgbm_num_boost_round, seed=seed)
    if backend == "torch_mlp":
        return _TorchMLPMultiOutputRegressor(seed=seed)
    raise ValueError(f"Unknown backup regressor backend {backend!r}.")


class _TorchMLPMultiOutputRegressor:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._ridge = _RidgeMultiOutputRegressor(alpha=1e-6)

    def fit(self, x: Array, y: Array, *, sample_weight: Optional[Array] = None) -> "_TorchMLPMultiOutputRegressor":
        warnings.warn(
            "torch_mlp backup regressor is not yet implemented; using ridge fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        self._ridge.fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x: Array) -> Array:
        return self._ridge.predict(x)


def _fit_truncated_svd(
    wc: Array,
    *,
    rank: int,
    sample_weight: Optional[Array],
    backend: str,
    seed: int,
) -> tuple[Array, Array, Array]:
    wc = _as_2d(wc, "wc").astype(np.float64, copy=False)
    weights = _normal_row_weights(sample_weight, wc.shape[0])
    weighted = wc * np.sqrt(weights)[:, None]
    rank = min(max(1, int(rank)), min(weighted.shape))
    if backend == "torch":
        try:
            import torch

            with torch.no_grad():
                u, s, vt = torch.linalg.svd(torch.as_tensor(weighted), full_matrices=False)
            singular_values = s.detach().cpu().numpy()[:rank]
            vt_np = vt.detach().cpu().numpy()[:rank]
        except Exception:
            _, singular_values, vt_np = _numpy_svd(weighted, rank)
    elif backend == "randomized" and rank < min(weighted.shape):
        _, singular_values, vt_np = _randomized_svd(weighted, rank=rank, seed=seed)
    else:
        _, singular_values, vt_np = _numpy_svd(weighted, rank)
    total = float(np.sum(np.linalg.svd(weighted, compute_uv=False) ** 2))
    evr = singular_values**2 / total if total > 0.0 else np.zeros_like(singular_values)
    return vt_np.astype(np.float64, copy=False), singular_values.astype(np.float64, copy=False), evr.astype(np.float64, copy=False)


def _numpy_svd(a: Array, rank: int) -> tuple[Array, Array, Array]:
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return u[:, :rank], s[:rank], vt[:rank]


def _randomized_svd(a: Array, *, rank: int, seed: int) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(int(seed))
    oversample = min(max(5, rank // 4), max(a.shape) if max(a.shape) else 5)
    q = min(a.shape[1], rank + oversample)
    omega = rng.normal(size=(a.shape[1], q))
    sample = a @ omega
    q_mat, _ = np.linalg.qr(sample, mode="reduced")
    b = q_mat.T @ a
    u_hat, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q_mat @ u_hat
    return u[:, :rank], s[:rank], vt[:rank]


def _candidate_ranks(config: FORITwoStageCVConfig, w_train: Array) -> list[int]:
    m = int(_as_2d(w_train, "w_train").shape[1])
    max_rank = min(int(config.max_rank), m)
    ranks = sorted({min(max(1, int(r)), max_rank) for r in config.low_rank_ranks if int(r) > 0})
    if config.low_rank_explained_variance is not None:
        wc = _as_2d(w_train, "w_train").astype(np.float64, copy=False)
        centered = wc - np.mean(wc, axis=0, keepdims=True)
        s = np.linalg.svd(centered, compute_uv=False)
        total = float(np.sum(s**2))
        if total > 0.0:
            cdf = np.cumsum(s**2) / total
            rank = int(np.searchsorted(cdf, float(config.low_rank_explained_variance)) + 1)
            ranks.append(min(rank, max_rank))
    return sorted(set(ranks or [min(max_rank, m)]))


def _build_data_adapter(
    *,
    states: Array,
    actions: Array,
    next_states: Array,
    target_actions: Optional[Array],
    target_next_actions: Optional[Array],
    target_policy: Any,
    action_space: Any,
    n_action_samples: int,
    gamma: float,
    episode_ids: Array,
    rewards: Optional[Array],
    initial_states: Optional[Array],
    initial_actions: Optional[Array],
    initial_weights: Optional[Array],
    initial_episode_ids: Optional[Array],
    terminated: Optional[Array],
    truncated: Optional[Array],
    done: Optional[Array],
    config: FORITwoStageCVConfig,
) -> _FORIData:
    s = _as_2d(states, "states").astype(np.float64, copy=False)
    a = _as_2d(actions, "actions").astype(np.float64, copy=False)
    sn = _as_2d(next_states, "next_states").astype(np.float64, copy=False)
    if not (s.shape[0] == a.shape[0] == sn.shape[0]):
        raise ValueError("states, actions, and next_states must have aligned rows.")
    ep = np.asarray(episode_ids).reshape(-1)
    if ep.shape[0] != s.shape[0]:
        raise ValueError("episode_ids must have the same number of rows as states.")
    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be in [0, 1).")
    ta = _as_2d(target_actions, "target_actions") if target_actions is not None else None
    if ta is None:
        ta = _as_2d(
            sample_target_successor_actions(s, target_policy, action_space, n_action_samples, config.seed),
            "target_actions",
        )
    tna = _as_2d(target_next_actions, "target_next_actions") if target_next_actions is not None else None
    if tna is None:
        tna = _as_2d(
            sample_target_successor_actions(sn, target_policy, action_space, n_action_samples, config.seed + 1),
            "target_next_actions",
        )
    if not (ta.shape[0] == s.shape[0] and tna.shape[0] == s.shape[0]):
        raise ValueError("target_actions and target_next_actions must align with states.")
    warnings_out: list[str] = []
    terminal_mode, sn, tna, continuation = _prepare_terminal_arrays(
        next_states=sn,
        target_next_actions=tna,
        terminated=terminated,
        truncated=truncated,
        done=done,
        config=config,
        warnings_out=warnings_out,
    )
    rewards_arr = None if rewards is None else np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards_arr is not None and rewards_arr.shape[0] != s.shape[0]:
        raise ValueError("rewards must align with states.")
    init_s = None if initial_states is None else _as_2d(initial_states, "initial_states").astype(np.float64, copy=False)
    init_a = None if initial_actions is None else _as_2d(initial_actions, "initial_actions").astype(np.float64, copy=False)
    init_w = None if initial_weights is None else np.asarray(initial_weights, dtype=np.float64).reshape(-1)
    init_ep = None if initial_episode_ids is None else np.asarray(initial_episode_ids).reshape(-1)
    return _FORIData(
        states=s,
        actions=a,
        next_states=sn,
        target_actions=ta.astype(np.float64, copy=False),
        target_next_actions=tna.astype(np.float64, copy=False),
        episode_ids=ep,
        gamma=float(gamma),
        rewards=rewards_arr,
        initial_states=init_s,
        initial_actions=init_a,
        initial_weights=init_w,
        initial_episode_ids=init_ep,
        continuation=continuation,
        terminal_mode=terminal_mode,
        warnings=warnings_out,
    )


def _prepare_terminal_arrays(
    *,
    next_states: Array,
    target_next_actions: Array,
    terminated: Optional[Array],
    truncated: Optional[Array],
    done: Optional[Array],
    config: FORITwoStageCVConfig,
    warnings_out: list[str],
) -> tuple[str, Array, Array, Array]:
    n = int(next_states.shape[0])
    terminal_indicator = _terminal_indicator(
        terminated=terminated,
        truncated=truncated,
        done=done,
        treat_timeouts_as_nonterminal=bool(config.treat_timeouts_as_nonterminal),
        n=n,
        warnings_out=warnings_out,
    )
    mode = str(config.terminal_mode)
    absorbing_available = config.absorbing_observation is not None and config.absorbing_action is not None
    if mode == "auto":
        mode = "absorbing_state" if absorbing_available else "live_only_submarkov"
        if not absorbing_available:
            message = "terminal_mode='auto' falling back to live_only_submarkov because no absorbing adapter was supplied."
            warnings_out.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
    if mode == "absorbing_state":
        if absorbing_available:
            terminal_mask = terminal_indicator > 0.0
            if np.any(terminal_mask):
                next_states = np.array(next_states, copy=True)
                target_next_actions = np.array(target_next_actions, copy=True)
                next_states[terminal_mask] = np.asarray(config.absorbing_observation, dtype=next_states.dtype).reshape(1, -1)
                target_next_actions[terminal_mask] = np.asarray(config.absorbing_action, dtype=target_next_actions.dtype).reshape(1, -1)
            continuation = np.ones(n, dtype=np.float64)
        elif np.any(terminal_indicator > 0.0):
            message = "terminal_mode='absorbing_state' requested without absorbing adapter; using live_only_submarkov."
            warnings_out.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
            mode = "live_only_submarkov"
            continuation = 1.0 - terminal_indicator
        else:
            continuation = np.ones(n, dtype=np.float64)
    else:
        continuation = 1.0 - terminal_indicator
    return mode, next_states, target_next_actions, np.clip(continuation, 0.0, 1.0).astype(np.float64, copy=False)


def _terminal_indicator(
    *,
    terminated: Optional[Array],
    truncated: Optional[Array],
    done: Optional[Array],
    treat_timeouts_as_nonterminal: bool,
    n: int,
    warnings_out: list[str],
) -> Array:
    term = None if terminated is None else _binary_vector(terminated, n, "terminated")
    trunc = None if truncated is None else _binary_vector(truncated, n, "truncated")
    done_arr = None if done is None else _binary_vector(done, n, "done")
    if term is not None:
        return term
    if done_arr is not None and trunc is not None:
        if treat_timeouts_as_nonterminal:
            return np.clip(done_arr * (1.0 - trunc), 0.0, 1.0)
        return done_arr
    if done_arr is not None:
        message = "done supplied without truncated; time-limit truncations may be treated as terminal."
        warnings_out.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return done_arr
    return np.zeros(n, dtype=np.float64)


def _binary_vector(values: Array, n: int, name: str) -> Array:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n):
        raise ValueError(f"{name} must have {n} rows.")
    return np.clip(arr, 0.0, 1.0)


def _resolve_initial_mode(data: _FORIData) -> str:
    if data.initial_states is None:
        return "none"
    if data.initial_actions is not None:
        return "joint"
    return "factored"


def _initial_subset(data: _FORIData, row_idx: Array) -> tuple[Optional[Array], Optional[Array], Optional[Array]]:
    if data.initial_states is None:
        return None, None, None
    rows = np.asarray(row_idx, dtype=np.int64).reshape(-1)
    allowed_eps = set(data.episode_ids[rows].tolist())
    n_init = int(data.initial_states.shape[0])
    if data.initial_episode_ids is not None:
        mask = np.isin(data.initial_episode_ids, list(allowed_eps))
    elif n_init == data.n:
        mask = np.zeros(data.n, dtype=bool)
        mask[rows] = True
    else:
        unique = np.unique(data.episode_ids)
        if n_init == unique.shape[0]:
            mask = np.isin(unique, list(allowed_eps))
        else:
            mask = np.ones(n_init, dtype=bool)
    s = data.initial_states[mask]
    a = None if data.initial_actions is None else data.initial_actions[mask]
    w = None if data.initial_weights is None else _normalize_numerator_weights(data.initial_weights[mask])
    return s, a, w


def _normalize_numerator_weights(weights: Array) -> Array:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(w, 0.0, out=w)
    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(w)
    return w * (w.shape[0] / total)


def _continuation_weights_for_density(data: _FORIData, idx: Array) -> Optional[Array]:
    if data.terminal_mode != "live_only_submarkov":
        return None
    return np.asarray(data.continuation[np.asarray(idx, dtype=np.int64)], dtype=np.float64).reshape(-1)


def _submarkov_density_ratio_score(pred_den: Array, pred_num: Array, weights: Array) -> float:
    den = np.asarray(pred_den, dtype=np.float64).reshape(-1)
    num = np.asarray(pred_num, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if num.shape[0] != w.shape[0]:
        raise ValueError("weights must align with numerator predictions.")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(w, 0.0, out=w)
    return float(np.mean(den**2) - 2.0 * np.mean(w * num))


def _backup_sample_weight(data: _FORIData, idx: Array) -> Optional[Array]:
    return _continuation_weights_for_density(data, idx)


def _first_stage_configs(config: FORITwoStageCVConfig) -> list[Any]:
    if config.first_stage_density_ratio_configs:
        return list(config.first_stage_density_ratio_configs)
    loss = "lsif" if config.first_stage_density_ratio_loss == "ulsif" else str(config.first_stage_density_ratio_loss)
    base = SourceStateRatioConfig(
        density_ratio_loss=loss,
        show_progress=False,
        num_boost_round=80,
        early_stopping_rounds=5,
        lgb_params={"verbose": -1, "num_threads": 1},
    )
    return [base]


def _default_candidate_specs() -> list[FORICandidateSpec]:
    return [
        FORICandidateSpec(
            candidate_id="boosted_stable_default",
            family="boosted",
            occupancy=OccupancyRegressionConfig.stable_defaults(show_progress=False)
            if hasattr(OccupancyRegressionConfig, "stable_defaults")
            else OccupancyRegressionConfig(show_progress=False),
        )
    ]


def _predict_candidate(spec: FORICandidateSpec, states: Array, actions: Array) -> Array:
    if spec.predictor is not None:
        return np.asarray(spec.predictor(states, actions), dtype=np.float64).reshape(-1)
    model = spec.model
    if model is None:
        raise ValueError(f"Candidate {spec.candidate_id!r} has no predictor or model.")
    try:
        import torch

        with torch.no_grad():
            return _predict_candidate_model(model, states, actions)
    except Exception:
        return _predict_candidate_model(model, states, actions)


def _predict_candidate_model(model: Any, states: Array, actions: Array) -> Array:
    if hasattr(model, "predict_ratio"):
        return np.asarray(model.predict_ratio(states, actions), dtype=np.float64).reshape(-1)
    if hasattr(model, "predict_state_action_ratio"):
        return np.asarray(model.predict_state_action_ratio(states, actions, clip=True), dtype=np.float64).reshape(-1)
    if callable(model):
        return np.asarray(model(states, actions), dtype=np.float64).reshape(-1)
    raise ValueError("Candidate model must expose predict_ratio, predict_state_action_ratio, or be callable.")


def _predict_internal_backup(internal: Any, x_current: Array, states: Array, actions: Array) -> Array:
    if hasattr(internal, "predict"):
        return np.asarray(internal.predict(x_current), dtype=np.float64).reshape(-1)
    if callable(internal):
        try:
            return np.asarray(internal(x_current), dtype=np.float64).reshape(-1)
        except TypeError:
            return np.asarray(internal(states, actions), dtype=np.float64).reshape(-1)
    raise ValueError("Internal backup model must expose predict(X) or be callable.")


def _cached_prediction_for_split(spec: FORICandidateSpec, split_name: str, n_rows: int) -> Optional[Array]:
    cached = spec.cached_predictions
    if cached is None:
        return None
    if isinstance(cached, Mapping):
        if split_name not in cached:
            return None
        arr = np.asarray(cached[split_name], dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(cached, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"Cached predictions for {spec.candidate_id!r}/{split_name} must have {n_rows} rows.")
    return arr


def _candidate_diagnostics(
    w_score: Array,
    *,
    action_shift: Optional[Mapping[str, float]] = None,
    config: Optional[FORITwoStageCVConfig] = None,
) -> dict[str, Array]:
    w = _as_2d(w_score, "w_score").astype(np.float64, copy=False)
    positive = np.maximum(w, 0.0)
    denom = np.sum(positive**2, axis=0)
    ess = np.divide(np.sum(positive, axis=0) ** 2, denom, out=np.zeros(w.shape[1], dtype=np.float64), where=denom > 0)
    mean = np.mean(w, axis=0)
    sd = np.std(w, axis=0)
    ratio_cv = np.divide(sd, np.abs(mean), out=np.full(w.shape[1], float("inf"), dtype=np.float64), where=np.abs(mean) > 1e-12)
    shift_l2 = float((action_shift or {}).get("policy_action_shift_l2", float("nan")))
    if config is None:
        near_uniform = np.zeros(w.shape[1], dtype=np.float64)
    else:
        near_uniform = (
            (ess / max(w.shape[0], 1) >= float(config.collapse_diagnostic_ess_fraction))
            & (ratio_cv <= float(config.collapse_diagnostic_weight_cv))
            & np.isfinite(shift_l2)
            & (shift_l2 > float(config.collapse_diagnostic_min_action_shift))
        ).astype(np.float64)
    return {
        "mean_ratio": mean,
        "second_moment": np.mean(w**2, axis=0),
        "negative_mass": np.mean(np.minimum(w, 0.0) ** 2, axis=0),
        "max_ratio": np.max(w, axis=0),
        "q50_ratio": np.quantile(w, 0.50, axis=0),
        "q90_ratio": np.quantile(w, 0.90, axis=0),
        "q95_ratio": np.quantile(w, 0.95, axis=0),
        "q99_ratio": np.quantile(w, 0.99, axis=0),
        "ESS": ess,
        "ESS_fraction": ess / max(w.shape[0], 1),
        "clipping_rate": np.zeros(w.shape[1], dtype=np.float64),
        "weight_cv": ratio_cv,
        "near_uniform_collapse": near_uniform,
        "policy_action_shift_l2": np.full(w.shape[1], shift_l2, dtype=np.float64),
        "policy_action_shift_mean_abs": np.full(
            w.shape[1],
            float((action_shift or {}).get("policy_action_shift_mean_abs", float("nan"))),
            dtype=np.float64,
        ),
    }


def _policy_action_shift(data: _FORIData, idx: Array) -> dict[str, float]:
    rows = np.asarray(idx, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        return {"policy_action_shift_l2": float("nan"), "policy_action_shift_mean_abs": float("nan")}
    logged = _as_2d(data.actions[rows], "actions").astype(np.float64, copy=False)
    target = _as_2d(data.target_actions[rows], "target_actions").astype(np.float64, copy=False)
    if logged.shape != target.shape:
        return {"policy_action_shift_l2": float("nan"), "policy_action_shift_mean_abs": float("nan")}
    diff = target - logged
    return {
        "policy_action_shift_l2": float(np.mean(np.linalg.norm(diff, axis=1))),
        "policy_action_shift_mean_abs": float(np.mean(np.abs(diff))),
    }


def _final_scores(abe: Array, diagnostics: Mapping[str, Array], config: FORITwoStageCVConfig) -> Array:
    out = np.asarray(abe, dtype=np.float64).copy()
    lambda_negative = float(config.lambda_negative)
    if lambda_negative > 0.0:
        out += lambda_negative * np.asarray(diagnostics["negative_mass"], dtype=np.float64)
    if float(config.lambda_second_moment) > 0.0:
        out += float(config.lambda_second_moment) * np.asarray(diagnostics["second_moment"], dtype=np.float64)
    if bool(config.enforce_mean_one_ratio) or float(config.lambda_mean_one) > 0.0:
        out += float(config.lambda_mean_one) * (
            np.asarray(diagnostics["mean_ratio"], dtype=np.float64) - float(config.expected_ratio_mass)
        ) ** 2
    return out


def _bootstrap_score_ses(
    *,
    residual: Array,
    w_score: Array,
    episode_ids: Array,
    final_score_components: Mapping[str, Any],
    config: FORITwoStageCVConfig,
    selected_idx: int = -1,
) -> tuple[Array, Array, Array, dict[str, Array]]:
    residual = _as_2d(residual, "residual").astype(np.float64, copy=False)
    w = _as_2d(w_score, "w_score").astype(np.float64, copy=False)
    m = residual.shape[1]
    n_boot = int(config.n_bootstrap)
    if n_boot <= 1:
        nan = np.full(m, float("nan"), dtype=np.float64)
        return nan, nan, nan, {}
    eps = np.asarray(episode_ids).reshape(-1)
    unique = np.unique(eps)
    sums = np.zeros((unique.shape[0], m), dtype=np.float64)
    counts = np.zeros(unique.shape[0], dtype=np.float64)
    ratio_sum = np.zeros((unique.shape[0], m), dtype=np.float64)
    ratio_sq_sum = np.zeros((unique.shape[0], m), dtype=np.float64)
    neg_sum = np.zeros((unique.shape[0], m), dtype=np.float64)
    for pos, ep in enumerate(unique):
        mask = eps == ep
        counts[pos] = float(np.sum(mask))
        sums[pos] = np.sum(residual[mask] ** 2, axis=0)
        ratio_sum[pos] = np.sum(w[mask], axis=0)
        ratio_sq_sum[pos] = np.sum(w[mask] ** 2, axis=0)
        neg_sum[pos] = np.sum(np.minimum(w[mask], 0.0) ** 2, axis=0)
    rng = np.random.default_rng(int(config.seed) + 505_051)
    abe_boot = np.zeros((n_boot, m), dtype=np.float64)
    final_boot = np.zeros((n_boot, m), dtype=np.float64)
    mean_boot = np.zeros((n_boot, m), dtype=np.float64)
    second_boot = np.zeros((n_boot, m), dtype=np.float64)
    neg_boot = np.zeros((n_boot, m), dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, unique.shape[0], size=unique.shape[0])
        count = float(np.sum(counts[draw]))
        count = max(count, 1.0)
        abe_b = np.sum(sums[draw], axis=0) / count
        mean_b = np.sum(ratio_sum[draw], axis=0) / count
        second_b = np.sum(ratio_sq_sum[draw], axis=0) / count
        neg_b = np.sum(neg_sum[draw], axis=0) / count
        final_b = abe_b.copy()
        if float(config.lambda_negative) > 0.0:
            final_b += float(config.lambda_negative) * neg_b
        if float(config.lambda_second_moment) > 0.0:
            final_b += float(config.lambda_second_moment) * second_b
        if bool(config.enforce_mean_one_ratio) or float(config.lambda_mean_one) > 0.0:
            final_b += float(config.lambda_mean_one) * (mean_b - float(config.expected_ratio_mass)) ** 2
        abe_boot[b] = abe_b
        final_boot[b] = final_b
        mean_boot[b] = mean_b
        second_boot[b] = second_b
        neg_boot[b] = neg_b
    del final_score_components
    paired_diff_se = np.full(m, float("nan"), dtype=np.float64)
    if 0 <= int(selected_idx) < m:
        diff_boot = final_boot - final_boot[:, [int(selected_idx)]]
        paired_diff_se = np.std(diff_boot, axis=0, ddof=1)
    return (
        np.std(abe_boot, axis=0, ddof=1),
        np.std(final_boot, axis=0, ddof=1),
        paired_diff_se,
        {
            "mean_ratio_se": np.std(mean_boot, axis=0, ddof=1),
            "second_moment_se": np.std(second_boot, axis=0, ddof=1),
            "negative_mass_se": np.std(neg_boot, axis=0, ddof=1),
        },
    )


def _one_se_selection(
    final_score: Array,
    final_se: Array,
    candidates: Sequence[FORICandidateSpec],
    selected_idx: int,
    *,
    method: str = "marginal",
    paired_diff_se: Optional[Array] = None,
) -> int:
    if selected_idx < 0:
        return -1
    method = str(method)
    if method == "paired" and paired_diff_se is not None:
        diff_se = np.asarray(paired_diff_se, dtype=np.float64).reshape(-1)
        base = float(final_score[selected_idx])
        eligible = [
            idx
            for idx, value in enumerate(final_score)
            if np.isfinite(value)
            and idx < diff_se.shape[0]
            and (
                idx == selected_idx
                or (np.isfinite(diff_se[idx]) and float(value) - base <= float(diff_se[idx]))
            )
        ]
    else:
        threshold = float(final_score[selected_idx])
        if np.isfinite(final_se[selected_idx]):
            threshold += float(final_se[selected_idx])
        eligible = [idx for idx, value in enumerate(final_score) if np.isfinite(value) and float(value) <= threshold]
    if not eligible:
        return selected_idx
    return min(eligible, key=lambda idx: _complexity_key(candidates[idx], idx))


def _complexity_key(spec: FORICandidateSpec, idx: int) -> tuple[Any, ...]:
    if spec.complexity_order_key is not None:
        key = spec.complexity_order_key
        return tuple(key) if isinstance(key, (tuple, list)) else (key,)
    iteration = math.inf if spec.iteration is None else int(spec.iteration)
    hp = dict(spec.hyperparams or {})
    regularization = -float(hp.get("regularization", hp.get("lambda_l2", hp.get("weight_decay", 0.0)) or 0.0))
    clip = float(hp.get("clip_upper", hp.get("occupancy_ratio_max", math.inf)) or math.inf)
    damping = -float(spec.damping_alpha if spec.damping_alpha is not None else hp.get("damping_alpha", 1.0))
    return (iteration, regularization, clip, damping, idx)


def _candidate_result_row(
    *,
    spec: FORICandidateSpec,
    index: int,
    abe: Array,
    abe_se: Array,
    final_score: Array,
    final_score_se: Array,
    paired_diff_se: Array,
    diagnostics: Mapping[str, Array],
    diagnostic_se: Mapping[str, Array],
    direct_abe: Optional[Array],
    naive_abe: Optional[Array],
    w_score_matrix: Array,
    selected_idx: int,
    one_se_idx: int,
    one_se_marginal_idx: int,
    one_se_paired_idx: int,
    rank_used: int,
    adjoint_regression_val_mse: float,
    backup_abe_val_score: float,
    direct_agreement_val_mse: float,
    rank_selection_score: float,
    rank_selection_metric: str,
    rank_selection_table: Sequence[Mapping[str, Any]],
    data: _FORIData,
    score_idx: Array,
    runtime_seconds: float,
    config: FORITwoStageCVConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": str(spec.candidate_id),
        "iteration": float("nan") if spec.iteration is None else int(spec.iteration),
        "hyperparams_summary": json.dumps(dict(spec.hyperparams or {}), sort_keys=True, default=str),
        "first_stage_omega0_id": "selected",
        "first_stage_cpi_id": "selected_direct",
        "ABE_score": float(abe[index]),
        "ABE_score_se": float(abe_se[index]) if np.size(abe_se) else float("nan"),
        "final_score": float(final_score[index]),
        "final_score_se": float(final_score_se[index]) if np.size(final_score_se) else float("nan"),
        "final_score_paired_diff_se": float(paired_diff_se[index]) if np.size(paired_diff_se) else float("nan"),
        "final_score_diff_to_min": float(final_score[index] - final_score[selected_idx]) if selected_idx >= 0 else float("nan"),
        "selected_min_score": bool(index == selected_idx),
        "selected_one_se": bool(index == one_se_idx),
        "selected_one_se_marginal": bool(index == one_se_marginal_idx),
        "selected_one_se_paired": bool(index == one_se_paired_idx),
        "one_se_method": str(config.one_se_method),
        "rank_used": int(rank_used),
        "adjoint_regression_val_mse": float(adjoint_regression_val_mse),
        "backup_abe_val_score": float(backup_abe_val_score),
        "direct_agreement_val_mse": float(direct_agreement_val_mse),
        "rank_selection_score": float(rank_selection_score),
        "rank_selection_metric": str(rank_selection_metric),
        "rank_selection_table": json.dumps(list(rank_selection_table), sort_keys=True, default=str),
        "direct_multioutput_ABE": float("nan") if direct_abe is None else float(direct_abe[index]),
        "naive_internal_ABE": float("nan") if naive_abe is None else float(naive_abe[index]),
        "projection_type": str(spec.projection_type),
        "damping_alpha": float("nan") if spec.damping_alpha is None else float(spec.damping_alpha),
        "complexity_order_key": repr(spec.complexity_order_key),
        "n_score_episodes": int(np.unique(data.episode_ids[score_idx]).shape[0]),
        "n_score_transitions": int(np.asarray(score_idx).shape[0]),
        "runtime_seconds": float(runtime_seconds),
        "peak_memory_mb": float("nan"),
    }
    for key, values in diagnostics.items():
        row[key] = float(np.asarray(values, dtype=np.float64).reshape(-1)[index])
    for key, values in diagnostic_se.items():
        row[key] = float(np.asarray(values, dtype=np.float64).reshape(-1)[index])
    if data.rewards is not None and bool(config.report_value_estimates):
        rewards = np.asarray(data.rewards, dtype=np.float64).reshape(-1)[score_idx]
        weights = _as_2d(w_score_matrix, "w_score_matrix")[:, index]
        value_norm = float(np.mean(weights * rewards)) if rewards.size else float("nan")
        row["value_estimate_normalized_occupancy"] = value_norm
        row["value_estimate_standard_return"] = value_norm / max(1.0 - float(data.gamma), 1e-12)
        row["value_estimate_se"] = float("nan")
    return row


def _weighted_matrix_mse(a: Array, b: Array, *, sample_weight: Optional[Array]) -> float:
    a = _as_2d(a, "a").astype(np.float64, copy=False)
    b = _as_2d(b, "b").astype(np.float64, copy=False)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    err = (a - b) ** 2
    if sample_weight is None:
        return float(np.mean(err))
    w = _normal_row_weights(sample_weight, a.shape[0])
    return float(np.sum(w[:, None] * err) / (np.sum(w) * a.shape[1]))


def _normal_row_weights(sample_weight: Optional[Array], n: int) -> Array:
    if sample_weight is None:
        return np.ones(int(n), dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != int(n):
        raise ValueError("sample_weight must align with rows.")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(w, 0.0, out=w)
    if float(np.sum(w)) <= 0.0:
        return np.ones(int(n), dtype=np.float64)
    return w


def _kfold_indices_by_episode_ids(episode_ids: Array, allowed_idx: Array, k: int, seed: int) -> list[Array]:
    allowed_idx = np.asarray(allowed_idx, dtype=np.int64).reshape(-1)
    groups = np.asarray(episode_ids).reshape(-1)
    unique = np.unique(groups[allowed_idx])
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(unique.shape[0])]
    fold_eps = np.array_split(shuffled, int(k))
    return [
        allowed_idx[np.isin(groups[allowed_idx], eps)].astype(np.int64, copy=False)
        for eps in fold_eps
        if eps.shape[0] > 0
    ]


def _indices_difference(base: Array, remove: Array) -> Array:
    base = np.asarray(base, dtype=np.int64).reshape(-1)
    remove_set = set(np.asarray(remove, dtype=np.int64).reshape(-1).tolist())
    return np.asarray([idx for idx in base.tolist() if idx not in remove_set], dtype=np.int64)


def _assert_split_disjoint(split_episode_ids: Mapping[str, Array]) -> None:
    names = list(split_episode_ids)
    for i, left in enumerate(names):
        left_set = set(np.asarray(split_episode_ids[left]).reshape(-1).tolist())
        for right in names[i + 1 :]:
            overlap = left_set.intersection(np.asarray(split_episode_ids[right]).reshape(-1).tolist())
            if overlap:
                raise ValueError(f"Episode leakage between {left} and {right}: {sorted(overlap)[:5]}")


def _as_2d(x: Any, name: str) -> Array:
    if x is None:
        raise ValueError(f"{name} is required.")
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must be a 1D or 2D array.")


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("inf")


def _cv_row(stage: str, config_id: int, fold: int, score: float, config: Any, mode: str, error: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "candidate_id": f"{stage}_{config_id:03d}",
        "config_id": int(config_id),
        "fold": int(fold),
        "mode": str(mode),
        "score": float(score),
        "error": str(error),
        "config": _config_dict(config),
    }


def _config_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        out = {}
        for key in config.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(config, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            elif isinstance(value, Mapping):
                out[key] = dict(value)
            elif isinstance(value, (tuple, list)):
                out[key] = list(value)
            else:
                out[key] = repr(value)
        return out
    return {"repr": repr(config)}


def _maybe_memmap_path(config: FORITwoStageCVConfig, name: str) -> Optional[Path]:
    if config.prediction_memmap_dir is None:
        return None
    root = Path(config.prediction_memmap_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}_{int(time.time() * 1e6)}.dat"


def _discrete_action_count(action_space: Any) -> Optional[int]:
    if action_space is None:
        return None
    if isinstance(action_space, int):
        return int(action_space)
    if hasattr(action_space, "n"):
        return int(action_space.n)
    return None


def _target_policy_probabilities(target_policy: Any, obs: Array) -> Optional[Array]:
    if hasattr(target_policy, "action_probabilities"):
        return np.asarray(target_policy.action_probabilities(obs), dtype=np.float64)
    if hasattr(target_policy, "predict_proba"):
        return np.asarray(target_policy.predict_proba(obs), dtype=np.float64)
    return None


def _sample_policy_once(target_policy: Any, obs: Array, rng: np.random.Generator) -> Array:
    if hasattr(target_policy, "sample_action"):
        try:
            return np.asarray(target_policy.sample_action(obs, rng), dtype=np.float64)
        except TypeError:
            return np.asarray(target_policy.sample_action(obs), dtype=np.float64)
    if hasattr(target_policy, "sample"):
        try:
            return np.asarray(target_policy.sample(obs, rng), dtype=np.float64)
        except TypeError:
            return np.asarray(target_policy.sample(obs), dtype=np.float64)
    if hasattr(target_policy, "predict"):
        return np.asarray(target_policy.predict(obs), dtype=np.float64)
    if callable(target_policy):
        try:
            return np.asarray(target_policy(obs, rng), dtype=np.float64)
        except TypeError:
            return np.asarray(target_policy(obs), dtype=np.float64)
    raise ValueError("target_policy must be callable or expose sample_action/sample/predict.")


__all__ = [
    "FORITwoStageCV",
    "FirstStageDensityRatioCV",
    "LowRankAdjointBellmanCV",
    "LowRankAdjointBackupRegressor",
    "DirectMultiOutputAdjointBackupRegressor",
    "FORIModelSelectionResult",
    "FORITwoStageCVConfig",
    "FORICandidateSpec",
    "FirstStageDensityRatioCVResult",
    "compute_candidate_ratio_matrix",
    "adjoint_bellman_residual",
    "sample_target_successor_actions",
    "split_by_episode_ids",
    "kfold_by_episode_ids",
    "load_fori_two_stage_config",
]
