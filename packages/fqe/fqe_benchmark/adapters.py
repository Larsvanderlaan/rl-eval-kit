from __future__ import annotations

from importlib import import_module
import time
from typing import Any

import numpy as np

from fqe_benchmark.types import BenchmarkConfig, BenchmarkDataset, EstimatorPreflight, FittedEstimator
from fqe import (
    BoostedFQEConfig,
    FQESearchSpace,
    FQETuningConfig,
    GoogleDualDICEConfig,
    NeuralFQEConfig,
    StationaryWeightedFQEConfig,
    fit_stationary_weighted_fqe,
    fit_fqe_lgbm,
    fit_fqe_neural,
    preflight_google_dualdice,
    preflight_minimax_weight,
    tune_fqe_auto,
)


Array = np.ndarray


def estimator_registry() -> dict[str, object]:
    adapters = [
        OursBoostedFQEAdapter(),
        OursBoostedFQETunedAdapter(),
        OursBoostedFQEStagedCVAdapter(),
        OursStationaryWeightedFQEAdapter(),
        OursGoogleDualDICEWeightedFQEAdapter(),
        OursMinimaxWeightedFQEAdapter(),
        OursGoogleDICERLExactWeightedFQEAdapter(),
        OursGoogleDICERLRecommendedWeightedFQEAdapter(),
        OursScopeRLMinimaxStateActionWeightedFQEAdapter(),
        OursScopeRLMinimaxStateWeightedFQEAdapter(),
        OursNeuralFQEAdapter(),
        OursNeuralFQETunedAdapter(),
        OursNeuralFQEStagedCVAdapter(),
        OursStationaryWeightedNeuralFQEAdapter(),
        OursStationaryWeightedNeuralFQEGoogleParityAdapter(),
        OursGoogleDualDICEWeightedNeuralFQEAdapter(),
        OursMinimaxWeightedNeuralFQEAdapter(),
        OursGoogleDICERLExactWeightedNeuralFQEAdapter(),
        OursGoogleDICERLRecommendedWeightedNeuralFQEAdapter(),
        OursScopeRLMinimaxStateActionWeightedNeuralFQEAdapter(),
        OursScopeRLMinimaxStateWeightedNeuralFQEAdapter(),
        LegacyBoostedFQEAdapter(),
        LegacyNeuralFQEAdapter(),
        ControlledLinearFQEAdapter(),
        D3RLPYFQEAdapter(),
        GooglePolicyEvalFQEAdapter(),
        DeepOPEReferenceFQEAdapter(),
    ]
    registry = {adapter.name: adapter for adapter in adapters}
    registry.update(
        {
            "boosted_fqe": registry["ours_boosted_fqe"],
            "boosted_fqe_stable": registry["ours_boosted_fqe"],
            "boosted_fqe_auto": registry["ours_boosted_fqe_tuned"],
            "boosted_fqe_staged_cv": registry["ours_boosted_fqe_staged_cv"],
            "stationary_weighted_fqe": registry["ours_google_dualdice_weighted_fqe"],
            "stationary_weighted_fqe_stable": registry["ours_google_dualdice_weighted_fqe"],
            "stationary_weighted_fori_fqe": registry["ours_stationary_weighted_fqe"],
            "stationary_weighted_fori_fqe_stable": registry["ours_stationary_weighted_fqe"],
            "google_dualdice_weighted_fqe": registry["ours_google_dualdice_weighted_fqe"],
            "stationary_weighted_google_dualdice_fqe": registry["ours_google_dualdice_weighted_fqe"],
            "stationary_weighted_minimax_fqe": registry["ours_minimax_weighted_fqe"],
            "stationary_weighted_google_dice_rl_dualdice_exact_fqe": registry[
                "ours_google_dice_rl_dualdice_exact_weighted_fqe"
            ],
            "stationary_weighted_google_dice_rl_recommended_fqe": registry[
                "ours_google_dice_rl_recommended_weighted_fqe"
            ],
            "stationary_weighted_scope_rl_minimax_state_action_fqe": registry[
                "ours_scope_rl_minimax_state_action_weighted_fqe"
            ],
            "stationary_weighted_scope_rl_minimax_state_fqe": registry[
                "ours_scope_rl_minimax_state_weighted_fqe"
            ],
            "neural_fqe": registry["ours_neural_fqe"],
            "neural_fqe_stable": registry["ours_neural_fqe"],
            "neural_fqe_auto": registry["ours_neural_fqe_tuned"],
            "neural_fqe_staged_cv": registry["ours_neural_fqe_staged_cv"],
            "stationary_weighted_neural_fqe": registry["ours_google_dualdice_weighted_neural_fqe"],
            "stationary_weighted_fori_neural_fqe": registry["ours_stationary_weighted_neural_fqe"],
            "stationary_weighted_neural_fqe_google_parity": registry[
                "ours_stationary_weighted_neural_fqe_google_parity"
            ],
            "google_dualdice_weighted_neural_fqe": registry["ours_google_dualdice_weighted_neural_fqe"],
            "stationary_weighted_google_dualdice_neural_fqe": registry["ours_google_dualdice_weighted_neural_fqe"],
            "stationary_weighted_minimax_neural_fqe": registry["ours_minimax_weighted_neural_fqe"],
            "stationary_weighted_google_dice_rl_dualdice_exact_neural_fqe": registry[
                "ours_google_dice_rl_dualdice_exact_weighted_neural_fqe"
            ],
            "stationary_weighted_google_dice_rl_recommended_neural_fqe": registry[
                "ours_google_dice_rl_recommended_weighted_neural_fqe"
            ],
            "stationary_weighted_scope_rl_minimax_state_action_neural_fqe": registry[
                "ours_scope_rl_minimax_state_action_weighted_neural_fqe"
            ],
            "stationary_weighted_scope_rl_minimax_state_neural_fqe": registry[
                "ours_scope_rl_minimax_state_weighted_neural_fqe"
            ],
        }
    )
    return registry


class OursBoostedFQEAdapter:
    name = "ours_boosted_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "array-based package FQE adapters are not enabled for Hopper placeholder rows; use hopper_fqe_benchmark.")
        try:
            import lightgbm  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "lightgbm is not installed.")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        model = fit_fqe_lgbm(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            config=_boosted_config(config, seed=seed, tuned=False),
        )
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics=dict(model.diagnostics))


class OursBoostedFQETunedAdapter(OursBoostedFQEAdapter):
    name = "ours_boosted_fqe_tuned"
    force_staged_cv = False

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        base = _boosted_config(config, seed=seed, tuned=True)
        tuned = tune_fqe_auto(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            families=("boosted",),
            search_space=FQESearchSpace(boosted=base),
            config=_automl_config(config, families=("boosted",), seed=seed, staged_cv=self.force_staged_cv),
        )
        runtime = time.perf_counter() - start
        model = tuned.model
        diagnostics = dict(model.diagnostics)
        diagnostics.update({"selected_candidate_id": tuned.selected_candidate_id, "selected_family": tuned.selected_family})
        return FittedEstimator(
            self.name,
            model,
            runtime,
            diagnostics=diagnostics,
            tuning_runtime_sec=runtime,
            tuning_rows=_tuning_rows(tuned),
        )


class OursBoostedFQEStagedCVAdapter(OursBoostedFQETunedAdapter):
    name = "ours_boosted_fqe_staged_cv"
    force_staged_cv = True


class OursStationaryWeightedFQEAdapter(OursBoostedFQEAdapter):
    name = "ours_stationary_weighted_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        base = super().preflight(config, dataset)
        if not base.available:
            return base
        try:
            import occupancy_ratio  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "occupancy-ratio is not installed.")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        result = fit_stationary_weighted_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            gamma_ratio=float(config.stationary_gamma_ratio),
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            config=StationaryWeightedFQEConfig(
                family="boosted",
                ratio_backend="occupancy_ratio",
                fqe_config=_boosted_config(config, seed=seed, tuned=False),
                **_stationary_boosted_ratio_configs(config, seed=seed),
            ),
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["estimator"] = self.name
        return FittedEstimator(self.name, result.fqe_model, time.perf_counter() - start, diagnostics=diagnostics)


class OursGoogleDualDICEWeightedFQEAdapter(OursBoostedFQEAdapter):
    name = "ours_google_dualdice_weighted_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        base = super().preflight(config, dataset)
        if not base.available:
            return base
        available, reason = preflight_google_dualdice(config.google_research_path)
        if not available:
            return EstimatorPreflight("missing_dependency", reason)
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        result = fit_stationary_weighted_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            gamma_ratio=float(config.stationary_gamma_ratio),
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            config=StationaryWeightedFQEConfig(
                family="boosted",
                ratio_backend="google_dualdice",
                fqe_config=_boosted_config(config, seed=seed, tuned=False),
                google_dualdice_config=_google_dualdice_config(config, seed=seed),
            ),
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["estimator"] = self.name
        return FittedEstimator(self.name, result.fqe_model, time.perf_counter() - start, diagnostics=diagnostics)


class OursMinimaxWeightedFQEAdapter(OursBoostedFQEAdapter):
    name = "ours_minimax_weighted_fqe"
    minimax_method = "auto"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        base = super().preflight(config, dataset)
        if not base.available:
            return base
        if self.minimax_method == "scope_rl_minimax_state" and dataset is not None and dataset.behavior_action_pscore is None:
            return EstimatorPreflight("missing_data", "SCOPE-RL state minimax weights require behavior_action_pscore.")
        try:
            minimax_config = _minimax_weight_config(config, seed=0, method=self.minimax_method)
            available, reason = preflight_minimax_weight(method=self.minimax_method, config=minimax_config)
        except ModuleNotFoundError as exc:
            return EstimatorPreflight("missing_dependency", f"occupancy-ratio minimax weights are unavailable: {exc}")
        except Exception as exc:
            return EstimatorPreflight("missing_dependency", f"minimax-weight preflight failed: {type(exc).__name__}: {exc}")
        if not available:
            return EstimatorPreflight("missing_dependency", reason)
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        result = fit_stationary_weighted_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            gamma_ratio=float(config.stationary_gamma_ratio),
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            config=StationaryWeightedFQEConfig(
                family="boosted",
                ratio_backend="minimax_weight",
                minimax_weight_method=self.minimax_method,
                minimax_weight_config=_minimax_weight_config(config, seed=seed, method=self.minimax_method),
                fqe_config=_boosted_config(config, seed=seed, tuned=False),
            ),
            episode_ids=dataset.episode_ids,
            timesteps=dataset.timesteps,
            step_per_trajectory=dataset.step_per_trajectory,
            behavior_action_pscore=dataset.behavior_action_pscore,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["estimator"] = self.name
        diagnostics["minimax_method"] = self.minimax_method
        return FittedEstimator(self.name, result.fqe_model, time.perf_counter() - start, diagnostics=diagnostics)


class OursGoogleDICERLExactWeightedFQEAdapter(OursMinimaxWeightedFQEAdapter):
    name = "ours_google_dice_rl_dualdice_exact_weighted_fqe"
    minimax_method = "google_dice_rl_dualdice_exact"


class OursGoogleDICERLRecommendedWeightedFQEAdapter(OursMinimaxWeightedFQEAdapter):
    name = "ours_google_dice_rl_recommended_weighted_fqe"
    minimax_method = "google_dice_rl_recommended"


class OursScopeRLMinimaxStateActionWeightedFQEAdapter(OursMinimaxWeightedFQEAdapter):
    name = "ours_scope_rl_minimax_state_action_weighted_fqe"
    minimax_method = "scope_rl_minimax_state_action"


class OursScopeRLMinimaxStateWeightedFQEAdapter(OursMinimaxWeightedFQEAdapter):
    name = "ours_scope_rl_minimax_state_weighted_fqe"
    minimax_method = "scope_rl_minimax_state"


class OursNeuralFQEAdapter:
    name = "ours_neural_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "array-based package neural FQE adapters are not enabled for Hopper placeholder rows; use hopper_fqe_benchmark.")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "torch is not installed.")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        _limit_torch_cpu_threads()
        start = time.perf_counter()
        model = fit_fqe_neural(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            config=_neural_config(config, seed=seed, tuned=False),
        )
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics=dict(model.diagnostics))


class OursNeuralFQETunedAdapter(OursNeuralFQEAdapter):
    name = "ours_neural_fqe_tuned"
    force_staged_cv = False

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        _limit_torch_cpu_threads()
        start = time.perf_counter()
        base = _neural_config(config, seed=seed, tuned=True)
        tuned = tune_fqe_auto(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            families=("neural",),
            search_space=FQESearchSpace(neural=base),
            config=_automl_config(config, families=("neural",), seed=seed, staged_cv=self.force_staged_cv),
        )
        runtime = time.perf_counter() - start
        model = tuned.model
        diagnostics = dict(model.diagnostics)
        diagnostics.update({"selected_candidate_id": tuned.selected_candidate_id, "selected_family": tuned.selected_family})
        return FittedEstimator(
            self.name,
            model,
            runtime,
            diagnostics=diagnostics,
            tuning_runtime_sec=runtime,
            tuning_rows=_tuning_rows(tuned),
        )


class OursNeuralFQEStagedCVAdapter(OursNeuralFQETunedAdapter):
    name = "ours_neural_fqe_staged_cv"
    force_staged_cv = True


class OursStationaryWeightedNeuralFQEAdapter(OursNeuralFQEAdapter):
    name = "ours_stationary_weighted_neural_fqe"
    ratio_preset = "stage_budget"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        base = super().preflight(config, dataset)
        if not base.available:
            return base
        try:
            import occupancy_ratio  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "occupancy-ratio is not installed.")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        _limit_torch_cpu_threads()
        start = time.perf_counter()
        result = fit_stationary_weighted_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            gamma_ratio=float(config.stationary_gamma_ratio),
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            config=StationaryWeightedFQEConfig(
                family="neural",
                ratio_backend="occupancy_ratio",
                neural_config=_neural_config(config, seed=seed, tuned=False),
                **_stationary_neural_ratio_configs(config, seed=seed, preset=self.ratio_preset),
            ),
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["estimator"] = self.name
        diagnostics["ratio_preset"] = self.ratio_preset
        return FittedEstimator(self.name, result.fqe_model, time.perf_counter() - start, diagnostics=diagnostics)


class OursStationaryWeightedNeuralFQEGoogleParityAdapter(OursStationaryWeightedNeuralFQEAdapter):
    name = "ours_stationary_weighted_neural_fqe_google_parity"
    ratio_preset = "google_parity"


class OursGoogleDualDICEWeightedNeuralFQEAdapter(OursNeuralFQEAdapter):
    name = "ours_google_dualdice_weighted_neural_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        base = super().preflight(config, dataset)
        if not base.available:
            return base
        available, reason = preflight_google_dualdice(config.google_research_path)
        if not available:
            return EstimatorPreflight("missing_dependency", reason)
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        _limit_torch_cpu_threads()
        start = time.perf_counter()
        result = fit_stationary_weighted_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            gamma_ratio=float(config.stationary_gamma_ratio),
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            config=StationaryWeightedFQEConfig(
                family="neural",
                ratio_backend="google_dualdice",
                neural_config=_neural_config(config, seed=seed, tuned=False),
                google_dualdice_config=_google_dualdice_config(config, seed=seed),
            ),
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["estimator"] = self.name
        return FittedEstimator(self.name, result.fqe_model, time.perf_counter() - start, diagnostics=diagnostics)


class OursMinimaxWeightedNeuralFQEAdapter(OursNeuralFQEAdapter):
    name = "ours_minimax_weighted_neural_fqe"
    minimax_method = "auto"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        base = super().preflight(config, dataset)
        if not base.available:
            return base
        if self.minimax_method == "scope_rl_minimax_state" and dataset is not None and dataset.behavior_action_pscore is None:
            return EstimatorPreflight("missing_data", "SCOPE-RL state minimax weights require behavior_action_pscore.")
        try:
            minimax_config = _minimax_weight_config(config, seed=0, method=self.minimax_method)
            available, reason = preflight_minimax_weight(method=self.minimax_method, config=minimax_config)
        except ModuleNotFoundError as exc:
            return EstimatorPreflight("missing_dependency", f"occupancy-ratio minimax weights are unavailable: {exc}")
        except Exception as exc:
            return EstimatorPreflight("missing_dependency", f"minimax-weight preflight failed: {type(exc).__name__}: {exc}")
        if not available:
            return EstimatorPreflight("missing_dependency", reason)
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        _limit_torch_cpu_threads()
        start = time.perf_counter()
        result = fit_stationary_weighted_fqe(
            states=dataset.states,
            actions=dataset.actions,
            next_states=dataset.next_states,
            target_actions=dataset.target_actions,
            next_actions=dataset.next_actions,
            rewards=dataset.rewards,
            gamma=dataset.gamma,
            gamma_ratio=float(config.stationary_gamma_ratio),
            terminals=dataset.terminals,
            sample_weight=dataset.sample_weight,
            initial_states=dataset.initial_states,
            initial_actions=dataset.initial_actions,
            config=StationaryWeightedFQEConfig(
                family="neural",
                ratio_backend="minimax_weight",
                minimax_weight_method=self.minimax_method,
                minimax_weight_config=_minimax_weight_config(config, seed=seed, method=self.minimax_method),
                neural_config=_neural_config(config, seed=seed, tuned=False),
            ),
            episode_ids=dataset.episode_ids,
            timesteps=dataset.timesteps,
            step_per_trajectory=dataset.step_per_trajectory,
            behavior_action_pscore=dataset.behavior_action_pscore,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["estimator"] = self.name
        diagnostics["minimax_method"] = self.minimax_method
        return FittedEstimator(self.name, result.fqe_model, time.perf_counter() - start, diagnostics=diagnostics)


class OursGoogleDICERLExactWeightedNeuralFQEAdapter(OursMinimaxWeightedNeuralFQEAdapter):
    name = "ours_google_dice_rl_dualdice_exact_weighted_neural_fqe"
    minimax_method = "google_dice_rl_dualdice_exact"


class OursGoogleDICERLRecommendedWeightedNeuralFQEAdapter(OursMinimaxWeightedNeuralFQEAdapter):
    name = "ours_google_dice_rl_recommended_weighted_neural_fqe"
    minimax_method = "google_dice_rl_recommended"


class OursScopeRLMinimaxStateActionWeightedNeuralFQEAdapter(OursMinimaxWeightedNeuralFQEAdapter):
    name = "ours_scope_rl_minimax_state_action_weighted_neural_fqe"
    minimax_method = "scope_rl_minimax_state_action"


class OursScopeRLMinimaxStateWeightedNeuralFQEAdapter(OursMinimaxWeightedNeuralFQEAdapter):
    name = "ours_scope_rl_minimax_state_weighted_neural_fqe"
    minimax_method = "scope_rl_minimax_state"


class LegacyBoostedFQEAdapter:
    name = "legacy_boosted_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "legacy boosted FQE adapter is not enabled for Hopper placeholder rows.")
        try:
            import_module("FQE.fqe_boosted")
        except Exception as exc:
            return EstimatorPreflight("missing_dependency", f"legacy boosted import failed: {type(exc).__name__}: {exc}")
        return EstimatorPreflight("ok")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        legacy = import_module("FQE.fqe_boosted")
        start = time.perf_counter()
        result = legacy.fit_fqe_boosted(
            S=dataset.states,
            A=dataset.actions,
            S_next=dataset.next_states,
            A_next=_flatten_next_actions(dataset.next_actions),
            rewards=dataset.rewards,
            discount_factor=dataset.gamma,
            is_terminal_outcome=dataset.terminals,
            weights=dataset.sample_weight,
            lgb_params={"num_iterations": 1, "learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5, "verbosity": -1},
            fit_control={"num_boost_rounds": max(4, min(config.boosted_num_iterations, 30)), "early_stopping_rounds": 5},
            seed=seed,
            verbose=False,
        )
        model = _LegacyBoostedModel(result["model"], dataset.state_dim, dataset.action_dim)
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics={"legacy_keys": sorted(result.keys())})


class LegacyNeuralFQEAdapter:
    name = "legacy_neural_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "legacy neural FQE adapter is not enabled for Hopper placeholder rows.")
        try:
            import_module("FQE.fqe_neural")
        except Exception as exc:
            return EstimatorPreflight("missing_dependency", f"legacy neural import failed: {type(exc).__name__}: {exc}")
        return EstimatorPreflight("unsupported_setting", "legacy neural FQE implementation in FQE/fqe_neural.py is value-only and does not expose Q(s,a) prediction.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("legacy_neural_fqe is unsupported for Q-mode benchmark datasets.")


class ControlledLinearFQEAdapter:
    name = "controlled_linear_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "controlled linear FQE is not enabled for Hopper placeholder rows.")
        if dataset is not None and dataset.domain not in {"controlled_synthetic", "tabular"}:
            return EstimatorPreflight("unsupported_setting", "controlled linear FQE is only enabled for truth-known datasets.")
        try:
            import_module("FQE_neurips.controlled_discounted_benchmark.fqe")
        except Exception as exc:
            return EstimatorPreflight(
                "ok",
                f"controlled benchmark import unavailable; using generic linear FQE fallback ({type(exc).__name__}: {exc}).",
            )
        return EstimatorPreflight("ok", "controlled benchmark import available; using generic linear adapter with matching feature-style baseline.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        start = time.perf_counter()
        model = _fit_linear_fqe(dataset, ridge=1e-4, n_iters=80 if config.stage == "smoke" else 200)
        return FittedEstimator(self.name, model, time.perf_counter() - start, diagnostics={"ridge": 1e-4})


class D3RLPYFQEAdapter:
    name = "d3rlpy_fqe"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "d3rlpy Hopper adapter is preflighted but not wired in v1.")
        try:
            import d3rlpy  # noqa: F401
        except ModuleNotFoundError:
            return EstimatorPreflight("missing_dependency", "d3rlpy is not installed.")
        return EstimatorPreflight("unsupported_setting", "d3rlpy adapter is preflighted but not enabled for synthetic array datasets in v1.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("d3rlpy adapter is not enabled for this v1 benchmark.")


class GooglePolicyEvalFQEAdapter:
    name = "google_policy_eval_fqe_l2"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        policy_eval = config.google_research_path / "policy_eval"
        if not policy_eval.exists():
            return EstimatorPreflight("missing_dependency", f"missing Google policy_eval at {policy_eval}.")
        if dataset is not None and dataset.domain == "offline_hopper":
            benchmark_dir = config.hopper_artifact_dir / "benchmark" / "dope"
            if not benchmark_dir.exists():
                return EstimatorPreflight("missing_data", f"missing Deep OPE benchmark artifacts at {benchmark_dir}.")
            return EstimatorPreflight("unsupported_setting", "Google policy_eval Hopper execution should be delegated to hopper_fqe_benchmark in v1.")
        return EstimatorPreflight("unsupported_setting", "Google policy_eval FQE-L2 is only enabled for Hopper/Deep OPE data.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("Google policy_eval FQE-L2 is not enabled for this dataset.")


class DeepOPEReferenceFQEAdapter:
    name = "deep_ope_reference_fqe_l2"

    def preflight(self, config: BenchmarkConfig, dataset: BenchmarkDataset | None = None) -> EstimatorPreflight:
        reference_path = config.hopper_artifact_dir / "benchmark" / "dope" / "d4rl_fqel2.pkl"
        if not reference_path.exists():
            return EstimatorPreflight("missing_data", f"missing Deep OPE reference file {reference_path}.")
        if dataset is not None and dataset.domain == "offline_hopper":
            return EstimatorPreflight("unsupported_setting", "Deep OPE reference file is present; reference-row materialization is delegated to hopper_fqe_benchmark in v1.")
        return EstimatorPreflight("unsupported_setting", "Deep OPE reference rows are only emitted for Hopper policy-ranking benchmarks.")

    def fit(self, dataset: BenchmarkDataset, config: BenchmarkConfig, seed: int) -> FittedEstimator:
        raise RuntimeError("Deep OPE reference rows are not fitted estimators.")


class _LegacyBoostedModel:
    def __init__(self, booster: Any, state_dim: int, action_dim: int) -> None:
        self.booster = booster
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    def predict_q(self, states: Array, actions: Array) -> Array:
        features = np.concatenate(
            [
                np.asarray(states, dtype=np.float64).reshape(-1, self.state_dim),
                np.asarray(actions, dtype=np.float64).reshape(-1, self.action_dim),
            ],
            axis=1,
        )
        return np.asarray(self.booster.predict(features), dtype=np.float64).reshape(-1)

    def estimate_policy_value(self, initial_states: Array, initial_actions: Array) -> float:
        return float(np.mean(self.predict_q(initial_states, initial_actions)))


class _LinearFQEModel:
    def __init__(self, theta: Array, state_dim: int, action_dim: int) -> None:
        self.theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    def predict_q(self, states: Array, actions: Array) -> Array:
        return _linear_features(states, actions) @ self.theta

    def estimate_policy_value(self, initial_states: Array, initial_actions: Array) -> float:
        return float(np.mean(self.predict_q(initial_states, initial_actions)))


def _boosted_config(config: BenchmarkConfig, *, seed: int, tuned: bool) -> BoostedFQEConfig:
    iterations = config.boosted_tune_num_iterations if tuned else config.boosted_num_iterations
    return BoostedFQEConfig.stable_defaults(
        num_iterations=int(iterations),
        validation_fraction=0.25,
        patience=5 if config.stage == "smoke" else 12,
        seed=int(seed),
        infer_value_bounds=True,
        show_progress=False,
        lgb_params={
            "learning_rate": 0.08,
            "num_leaves": 15 if config.stage == "smoke" else 31,
            "min_data_in_leaf": 5 if config.stage == "smoke" else 30,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "num_threads": 1,
        },
    )


def _stationary_boosted_ratio_configs(config: BenchmarkConfig, *, seed: int) -> dict[str, Any]:
    from occupancy_ratio import ActionRatioConfig, OccupancyRegressionConfig, SourceStateRatioConfig, TransitionRatioConfig

    smoke = config.stage == "smoke"
    nuisance_rounds = 4 if smoke else max(20, min(80, int(config.boosted_num_iterations)))
    return {
        "occupancy_config": OccupancyRegressionConfig(
            num_iterations=4 if smoke else max(20, min(80, int(config.boosted_num_iterations))),
            trees_per_iteration=1,
            mcmc_samples=4 if smoke else 24,
            batch_size=128 if smoke else 512,
            validation_fraction=0.25,
            patience=3 if smoke else 8,
            early_stopping=False,
            seed=int(seed) + 211,
            show_progress=False,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            lgb_params={
                "learning_rate": 0.08,
                "num_leaves": 15 if smoke else 31,
                "min_data_in_leaf": 2 if smoke else 20,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 1,
            },
        ),
        "action_ratio_config": ActionRatioConfig(
            num_boost_round=nuisance_rounds,
            validation_fraction=0.25,
            early_stopping_rounds=3 if smoke else 8,
            prediction_max=50.0,
            density_ratio_loss="lsif",
            show_progress=False,
            lgb_params={
                "learning_rate": 0.08,
                "num_leaves": 15 if smoke else 31,
                "min_data_in_leaf": 2 if smoke else 20,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 1,
            },
        ),
        "source_state_ratio_config": SourceStateRatioConfig(
            num_boost_round=nuisance_rounds,
            validation_fraction=0.25,
            early_stopping_rounds=3 if smoke else 8,
            prediction_max=50.0,
            density_ratio_loss="lsif",
            show_progress=False,
            lgb_params={
                "learning_rate": 0.08,
                "num_leaves": 15 if smoke else 31,
                "min_data_in_leaf": 2 if smoke else 20,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 1,
            },
        ),
        "transition_ratio_config": TransitionRatioConfig(
            num_boost_round=nuisance_rounds,
            permutation_samples=2 if smoke else 8,
            validation_fraction=0.25,
            early_stopping_rounds=3 if smoke else 8,
            prediction_max=50.0,
            density_ratio_loss="lsif",
            show_progress=False,
            lgb_params={
                "learning_rate": 0.08,
                "num_leaves": 15 if smoke else 31,
                "min_data_in_leaf": 2 if smoke else 20,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "num_threads": 1,
            },
        ),
    }


def _stationary_neural_ratio_configs(
    config: BenchmarkConfig,
    *,
    seed: int,
    preset: str = "stage_budget",
) -> dict[str, Any]:
    from occupancy_ratio import (
        NeuralActionRatioConfig,
        NeuralOccupancyRegressionConfig,
        NeuralSourceStateRatioConfig,
        NeuralTransitionRatioConfig,
    )

    smoke = config.stage == "smoke"
    if preset not in {"stage_budget", "google_parity"}:
        raise ValueError("stationary neural ratio preset must be 'stage_budget' or 'google_parity'.")
    hidden = (32, 32) if smoke else ((256, 256) if preset == "google_parity" else (64, 64))
    activation = "silu" if smoke or preset == "stage_budget" else "relu"
    action_steps = 80 if smoke else 800
    transition_steps = 80 if smoke else 1000
    occupancy_iterations = 8 if smoke else 60
    occupancy_gradient_steps = 4 if smoke else 6
    mcmc_samples = 8 if smoke else 24
    direct_adjoint_steps = 32 if smoke else 128
    learning_rate = 1e-3 if smoke else 5e-4
    batch_size = 256 if smoke else 512
    patience = 8 if smoke else 12
    return {
        "one_step_ratio_mode": "factored",
        "occupancy_config": NeuralOccupancyRegressionConfig.stable_defaults(
            hidden_dims=hidden,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=1e-5,
            batch_size=batch_size,
            num_iterations=occupancy_iterations,
            gradient_steps_per_iteration=occupancy_gradient_steps,
            mcmc_samples=mcmc_samples,
            validation_fraction=0.25,
            patience=5 if smoke else 12,
            validation_warmup_iterations=2,
            fixed_point_damping=0.5,
            normalize_occupancy=True,
            occupancy_ratio_max=50.0,
            clip_pseudo_outcomes=True,
            pseudo_outcome_upper_quantile=0.995,
            occupancy_sample_weight_mode="uniform",
            occupancy_sample_weight_max=20.0,
            direct_adjoint_steps=direct_adjoint_steps,
            direct_one_step_max_steps=transition_steps,
            direct_one_step_hidden_dims=hidden,
            direct_one_step_density_ratio_loss="lsif",
            direct_one_step_prediction_max=50.0,
            direct_one_step_moment_calibration="scalar",
            grad_clip_norm=5.0,
            device="cpu",
            seed=int(seed) + 811,
            show_progress=False,
        ),
        "action_ratio_config": NeuralActionRatioConfig.balanced_defaults(
            hidden_dims=hidden,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=1e-5,
            batch_size=batch_size,
            max_steps=action_steps,
            validation_fraction=0.25,
            patience=patience,
            prediction_max=50.0,
            density_ratio_loss="lsif",
            moment_calibration="scalar",
            device="cpu",
            seed=int(seed) + 821,
        ),
        "source_state_ratio_config": NeuralSourceStateRatioConfig.balanced_defaults(
            hidden_dims=hidden,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=1e-5,
            batch_size=batch_size,
            max_steps=action_steps,
            validation_fraction=0.25,
            patience=patience,
            prediction_max=50.0,
            density_ratio_loss="lsif",
            moment_calibration="scalar",
            device="cpu",
            seed=int(seed) + 831,
        ),
        "transition_ratio_config": NeuralTransitionRatioConfig.balanced_defaults(
            hidden_dims=hidden,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=1e-5,
            batch_size=batch_size,
            max_steps=transition_steps,
            permutation_samples=4 if smoke else 16,
            validation_fraction=0.25,
            patience=patience,
            prediction_max=50.0,
            density_ratio_loss="lsif",
            moment_calibration="scalar",
            device="cpu",
            seed=int(seed) + 841,
        ),
    }


def _automl_config(
    config: BenchmarkConfig,
    *,
    families: tuple[str, ...],
    seed: int,
    staged_cv: bool = False,
) -> FQETuningConfig:
    budget = str(config.automl_tuning)
    if budget == "off":
        budget = "balanced"
    return FQETuningConfig(
        families=families,
        cv_folds=3,
        seed=int(seed),
        budget=budget,
        max_candidates=8 if budget == "fast" or config.stage == "smoke" else 12,
        promotion_candidates=2 if budget == "fast" or config.stage == "smoke" else 4,
        refit=True,
        stable_fallback=True,
        staged_bootstrap_cv=bool(config.staged_cv or staged_cv),
        staged_cv_iterations=config.staged_cv_iterations,
    )


def _google_dualdice_config(config: BenchmarkConfig, *, seed: int) -> GoogleDualDICEConfig:
    return GoogleDualDICEConfig(
        google_research_path=config.google_research_path,
        num_updates=int(config.google_dualdice_num_updates),
        batch_size=int(config.google_dualdice_batch_size),
        seed=int(seed),
    )


def _minimax_weight_config(config: BenchmarkConfig, *, seed: int, method: str):
    from occupancy_ratio import (
        GoogleDICERLConfig,
        GoogleDualDICEConfig as OccupancyGoogleDualDICEConfig,
        MinimaxWeightConfig,
        ScopeRLMinimaxWeightConfig,
    )

    return MinimaxWeightConfig(
        method=method,
        google_policy_eval=OccupancyGoogleDualDICEConfig(
            google_research_path=config.google_research_path,
            num_updates=int(config.google_dualdice_num_updates),
            batch_size=int(config.google_dualdice_batch_size),
            seed=int(seed),
        ),
        google_dice_rl=GoogleDICERLConfig(
            dice_rl_repo_path=config.dice_rl_repo_path,
            num_steps=int(config.dice_rl_num_steps),
            batch_size=int(config.dice_rl_batch_size),
            learning_rate=float(config.dice_rl_learning_rate),
            hidden_dims=tuple(int(width) for width in config.dice_rl_hidden_dims),
            seed=int(seed),
        ),
        scope_rl=ScopeRLMinimaxWeightConfig(
            scope_rl_repo_path=config.scope_rl_repo_path,
            n_steps=int(config.scope_rl_n_steps),
            n_steps_per_epoch=int(config.scope_rl_n_steps_per_epoch),
            batch_size=int(config.scope_rl_batch_size),
            learning_rate=float(config.scope_rl_learning_rate),
            hidden_dim=int(config.scope_rl_hidden_dim),
            bandwidth=float(config.scope_rl_bandwidth),
            seed=int(seed),
            device="cpu",
        ),
    )


def _tuning_rows(tuned) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tuned.candidate_rows():
        out = dict(row)
        out["tuning_stage"] = "automl_candidate"
        rows.append(out)
    for row in tuned.fold_rows():
        out = dict(row)
        out["tuning_stage"] = "automl_fold"
        rows.append(out)
    staged_rows = tuned.staged_cv_rows() if hasattr(tuned, "staged_cv_rows") else []
    for row in staged_rows:
        out = dict(row)
        out["tuning_stage"] = "automl_staged_cv"
        out.setdefault("scoring", "staged_bootstrapped_loss")
        rows.append(out)
    return rows


def _limit_torch_cpu_threads(num_threads: int = 1) -> None:
    try:
        import torch
    except Exception:
        return
    try:
        torch.set_num_threads(int(num_threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(int(num_threads))
    except Exception:
        pass


def _neural_config(config: BenchmarkConfig, *, seed: int, tuned: bool) -> NeuralFQEConfig:
    iterations = config.neural_tune_num_iterations if tuned else config.neural_num_iterations
    steps = config.neural_tune_gradient_steps_per_iteration if tuned else config.neural_gradient_steps_per_iteration
    patience = 4 if config.stage == "smoke" else max(8, min(25, int(iterations) // 4))
    return NeuralFQEConfig.stable_defaults(
        hidden_dims=(32, 32) if config.stage == "smoke" else (256, 256),
        learning_rate=2e-3 if config.stage == "smoke" else 3e-4,
        weight_decay=1e-4,
        batch_size=64 if config.stage == "smoke" else 512,
        num_iterations=int(iterations),
        gradient_steps_per_iteration=int(steps),
        target_update_tau=0.5 if config.stage == "smoke" else 0.05,
        validation_fraction=0.25,
        patience=patience,
        min_improvement=1e-6,
        seed=int(seed),
        device="cpu",
        infer_value_bounds=True,
        show_progress=False,
    )


def _flatten_next_actions(next_actions: Array) -> Array:
    arr = np.asarray(next_actions, dtype=np.float64)
    if arr.ndim == 3:
        return arr[:, 0, :]
    return arr


def _linear_features(states: Array, actions: Array) -> Array:
    s = np.asarray(states, dtype=np.float64).reshape(states.shape[0], -1)
    a = np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1)
    x = np.concatenate([s, a], axis=1)
    quad = [x[:, i : i + 1] * x[:, j : j + 1] for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    return np.concatenate([np.ones((x.shape[0], 1)), x, *quad], axis=1)


def _fit_linear_fqe(dataset: BenchmarkDataset, *, ridge: float, n_iters: int) -> _LinearFQEModel:
    phi = _linear_features(dataset.states, dataset.actions)
    phi_next = _linear_features(dataset.next_states, _flatten_next_actions(dataset.next_actions))
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    weights = np.ones_like(rewards) if dataset.sample_weight is None else np.asarray(dataset.sample_weight, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 1e-12)
    gram = phi.T @ (weights[:, None] * phi) + float(ridge) * np.eye(phi.shape[1])
    reward_rhs = phi.T @ (weights * rewards)
    next_mat = phi.T @ (weights[:, None] * phi_next)
    theta = np.zeros(phi.shape[1], dtype=np.float64)
    for _ in range(int(n_iters)):
        rhs = reward_rhs + dataset.gamma * (next_mat @ theta)
        theta_new = np.linalg.solve(gram, rhs)
        if np.linalg.norm(theta_new - theta) <= 1e-10 * max(1.0, np.linalg.norm(theta)):
            theta = theta_new
            break
        theta = theta_new
    return _LinearFQEModel(theta=theta, state_dim=dataset.state_dim, action_dim=dataset.action_dim)
