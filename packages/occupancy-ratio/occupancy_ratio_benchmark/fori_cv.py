from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from occupancy_ratio.fit_occupancy_ratio import (
    ActionRatioConfig,
    OccupancyRegressionConfig,
    SourceStateRatioConfig,
    TransitionRatioConfig,
    _as_2d,
    _ess,
    _fold_initial_states,
    _fold_initial_weights,
    _make_stabilized_fixed_point_target,
    _predict_processed_source_state_ratio,
    fit_discounted_occupancy_ratio,
    make_direct_adjoint_occupancy_dataset,
    make_forward_occupancy_dataset,
)
from occupancy_ratio.fit_occupancy_ratio_neural import (
    NeuralActionRatioConfig,
    NeuralOccupancyRegressionConfig,
    NeuralSourceStateRatioConfig,
    NeuralTransitionRatioConfig,
    _NeuralDirectTargetBuilder,
    _NeuralTargetBuilder,
    fit_discounted_occupancy_ratio_neural,
)
from occupancy_ratio_benchmark.data import BenchmarkDataset


Array = np.ndarray


@dataclass(frozen=True)
class FORICVCandidate:
    """One leakage-safe FORI CV candidate."""

    name: str
    family: str = "neural"
    occupancy: OccupancyRegressionConfig | NeuralOccupancyRegressionConfig | None = None
    action_ratio: ActionRatioConfig | NeuralActionRatioConfig | None = None
    source_state_ratio: SourceStateRatioConfig | NeuralSourceStateRatioConfig | None = None
    transition_ratio: TransitionRatioConfig | NeuralTransitionRatioConfig | None = None
    initial_ratio_mode: str = "auto"
    one_step_ratio_mode: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FORICVFit:
    candidate: FORICVCandidate
    fold: int
    train_idx: Array
    model: Any
    runtime_sec: float


@dataclass
class FORICVResult:
    rows: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    selected_candidate: str | None
    plot_paths: list[Path] = field(default_factory=list)


def fit_fori_cv_candidate(
    dataset: BenchmarkDataset,
    candidate: FORICVCandidate,
    train_idx: Array,
    *,
    fold: int = 0,
    seed: int | None = None,
) -> FORICVFit:
    """Fit the full FORI pipeline using only the supplied training rows."""

    start = time.perf_counter()
    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_next = _as_2d(dataset.next_states, "next_states")
    A_pi = _as_2d(dataset.target_actions, "target_actions")
    A_next_pi = _as_2d(dataset.next_target_actions, "next_target_actions")
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    fit_seed = int(dataset.seed if seed is None else seed) + 10_003 * (int(fold) + 1)
    S_initial = _fold_initial_states(dataset.initial_states, train_idx, S.shape[0])
    A_initial = _fold_initial_states(dataset.initial_actions, train_idx, S.shape[0])
    initial_weights = _fold_initial_weights(dataset.initial_weights, dataset.initial_states, train_idx, S.shape[0])

    if str(candidate.family) == "boosted":
        occupancy = candidate.occupancy if candidate.occupancy is not None else OccupancyRegressionConfig()
        action_ratio = candidate.action_ratio if candidate.action_ratio is not None else ActionRatioConfig()
        source = candidate.source_state_ratio if candidate.source_state_ratio is not None else SourceStateRatioConfig()
        transition = candidate.transition_ratio if candidate.transition_ratio is not None else TransitionRatioConfig()
        model = fit_discounted_occupancy_ratio(
            states=S[train_idx],
            actions=A[train_idx],
            next_states=S_next[train_idx],
            target_actions=A_pi[train_idx],
            gamma=float(dataset.gamma),
            initial_states=S_initial,
            initial_actions=A_initial,
            initial_weights=initial_weights,
            target_next_actions=A_next_pi[train_idx],
            initial_ratio_mode=str(candidate.initial_ratio_mode),
            one_step_ratio_mode=str(candidate.one_step_ratio_mode),
            occupancy=replace(occupancy, seed=fit_seed, show_progress=False),
            action_ratio=replace(action_ratio, show_progress=False),
            source_state_ratio=replace(source, show_progress=False),
            transition_ratio=replace(transition, show_progress=False),
        )
    elif str(candidate.family) == "neural":
        _stabilize_torch_runtime()
        occupancy = candidate.occupancy if candidate.occupancy is not None else NeuralOccupancyRegressionConfig()
        action_ratio = candidate.action_ratio if candidate.action_ratio is not None else NeuralActionRatioConfig()
        source = candidate.source_state_ratio if candidate.source_state_ratio is not None else NeuralSourceStateRatioConfig()
        transition = candidate.transition_ratio if candidate.transition_ratio is not None else NeuralTransitionRatioConfig()
        model = fit_discounted_occupancy_ratio_neural(
            states=S[train_idx],
            actions=A[train_idx],
            next_states=S_next[train_idx],
            target_actions=A_pi[train_idx],
            gamma=float(dataset.gamma),
            initial_states=S_initial,
            initial_actions=A_initial,
            initial_weights=initial_weights,
            target_next_actions=A_next_pi[train_idx],
            initial_ratio_mode=str(candidate.initial_ratio_mode),
            one_step_ratio_mode=str(candidate.one_step_ratio_mode),
            occupancy=replace(occupancy, seed=fit_seed, show_progress=False),
            action_ratio=replace(action_ratio, seed=fit_seed + 101),
            source_state_ratio=replace(source, seed=fit_seed + 211),
            transition_ratio=replace(transition, seed=fit_seed + 307),
        )
    else:
        raise ValueError("candidate.family must be 'boosted' or 'neural'.")

    return FORICVFit(
        candidate=candidate,
        fold=int(fold),
        train_idx=train_idx,
        model=model,
        runtime_sec=float(time.perf_counter() - start),
    )


def score_fixed_point_residual(
    fit: FORICVFit,
    dataset: BenchmarkDataset,
    valid_idx: Array,
    *,
    seed: int = 123,
) -> dict[str, Any]:
    """Score held-out residual of the fitted adjoint update.

    The update is rebuilt from the fold-trained nuisance/source objects and the
    final train-fold weights. Validation rows are used only as query points.
    """

    valid_idx = np.asarray(valid_idx, dtype=np.int64).reshape(-1)
    train_idx = np.asarray(fit.train_idx, dtype=np.int64).reshape(-1)
    model = fit.model
    legacy = model.to_legacy_dict()
    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_next = _as_2d(dataset.next_states, "next_states")
    A_next_pi = _as_2d(dataset.next_target_actions, "next_target_actions")
    X_val = np.concatenate([S[valid_idx], A[valid_idx]], axis=1)
    current = np.asarray(model.predict_state_action_ratio(S[valid_idx], A[valid_idx], clip=True), dtype=np.float64)
    w_train = np.asarray(model.predict_state_action_ratio(S[train_idx], A[train_idx], clip=True), dtype=np.float64)
    source_val = _predict_source_weight(model, S[valid_idx], A[valid_idx])

    direct = bool(legacy.get("one_step_direct_ratio_enabled", False))
    if direct:
        c_val = _predict_one_step_ratio(model, X_val)
        if c_val is None:
            raw_update = (1.0 - float(dataset.gamma)) * source_val + float(dataset.gamma) * np.asarray(
                model.predict_state_action_ratio(S_next[valid_idx], A_next_pi[valid_idx], clip=True),
                dtype=np.float64,
            )
        elif str(fit.candidate.family) == "boosted":
            builder = make_direct_adjoint_occupancy_dataset(
                X_sa_successor=np.concatenate([S_next[train_idx], A_next_pi[train_idx]], axis=1),
                X_sa_query=X_val,
                c_ratio_query=c_val,
                w_source_query=source_val,
                gamma=float(dataset.gamma),
                seed=int(seed) + 4_991 * (fit.fold + 1),
                num_boost_round=max(1, int(legacy.get("direct_adjoint_num_boost_round") or 1)),
                loss=str(legacy.get("direct_adjoint_loss", legacy.get("loss", "squared"))),
                validation_fraction=0.0,
                early_stopping_rounds=0,
            )
            raw_update = np.asarray(builder(w_beh=w_train)["y"], dtype=np.float64)
        else:
            config = fit.candidate.occupancy
            if not isinstance(config, NeuralOccupancyRegressionConfig):
                config = NeuralOccupancyRegressionConfig()
            builder = _NeuralDirectTargetBuilder(
                X_sa_successor=np.concatenate([S_next[train_idx], A_next_pi[train_idx]], axis=1),
                X_sa_query=X_val,
                c_ratio_query=c_val,
                w_source_query=source_val,
                gamma=float(dataset.gamma),
                config=config,
                seed=int(seed) + 4_991 * (fit.fold + 1),
            )
            raw_update = np.asarray(builder(w_beh=w_train)["y"], dtype=np.float64)
    else:
        raw_update = _forward_adjoint_update(fit, dataset, valid_idx, w_train, seed=seed)

    update = _stabilize_update(raw_update, current, legacy)
    resid = current - update
    return {
        "fp": float(np.mean(resid**2)) if resid.size else float("nan"),
        "fp_abs_mean": float(np.mean(np.abs(resid))) if resid.size else float("nan"),
        "fp_rel_abs_mean": float(np.mean(np.abs(resid)) / (np.mean(np.abs(current)) + 1e-12)) if resid.size else float("nan"),
    }


def score_moment_balance(
    fit: FORICVFit,
    dataset: BenchmarkDataset,
    valid_idx: Array,
    *,
    seed: int = 123,
    rff_features: int = 16,
    raw_features: bool = True,
    reward_features: bool = True,
    reward_phi_features: bool = True,
    reward_hidden_dims: Sequence[int] = (64, 64),
    reward_max_steps: int = 200,
    reward_patience: int = 15,
    reward_feature_cap: int = 16,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Score held-out discounted occupancy moment balance."""

    valid_idx = np.asarray(valid_idx, dtype=np.int64).reshape(-1)
    train_idx = np.asarray(fit.train_idx, dtype=np.int64).reshape(-1)
    model = fit.model
    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_next = _as_2d(dataset.next_states, "next_states")
    A_next_pi = _as_2d(dataset.next_target_actions, "next_target_actions")
    init_idx = _initial_validation_indices(dataset, valid_idx, S.shape[0])
    S0 = _as_2d(dataset.initial_states, "initial_states")[init_idx]
    A0 = _as_2d(dataset.initial_actions, "initial_actions")[init_idx]

    features = _moment_features(
        train_states=S[train_idx],
        train_actions=A[train_idx],
        eval_states=S[valid_idx],
        eval_actions=A[valid_idx],
        next_states=S_next[valid_idx],
        next_actions=A_next_pi[valid_idx],
        initial_states=S0,
        initial_actions=A0,
        seed=int(seed) + 7_919 * (fit.fold + 1),
        rff_features=int(rff_features),
        raw_features=bool(raw_features),
        reward_block=None
        if not bool(reward_features)
        else _crossfit_reward_feature_block(
            dataset=dataset,
            train_idx=train_idx,
            valid_idx=valid_idx,
            initial_idx=init_idx,
            seed=int(seed) + 17_077 * (fit.fold + 1),
            hidden_dims=tuple(int(width) for width in reward_hidden_dims),
            max_steps=int(reward_max_steps),
            patience=int(reward_patience),
            feature_cap=int(reward_feature_cap),
            include_phi=bool(reward_phi_features),
        ),
    )
    weights = np.asarray(model.predict_state_action_ratio(S[valid_idx], A[valid_idx], clip=True), dtype=np.float64)
    valid_term = weights[:, None] * (features["eval"] - float(dataset.gamma) * features["next"])
    initial_term = (1.0 - float(dataset.gamma)) * features["initial"]
    deltas = np.mean(valid_term, axis=0) - np.mean(initial_term, axis=0)
    var_valid = np.var(valid_term, axis=0, ddof=1) / max(valid_term.shape[0], 1)
    var_initial = np.var(initial_term, axis=0, ddof=1) / max(initial_term.shape[0], 1)
    sigma2 = np.maximum(var_valid + var_initial, float(eps))
    z2 = deltas**2 / sigma2
    return {
        "mb": float(np.mean(z2)),
        "mb_max_z": float(np.sqrt(np.max(z2))) if z2.size else float("nan"),
        "mb_mean_abs_delta": float(np.mean(np.abs(deltas))) if deltas.size else float("nan"),
        "mb_features": int(z2.size),
        "mb_reward_features": int(features.get("reward_feature_count", 0)),
        **_observed_reward_diagnostic(dataset, valid_idx, weights),
    }


def score_value_grouped_moment_balance(
    fit: FORICVFit,
    dataset: BenchmarkDataset,
    valid_idx: Array,
    *,
    seed: int = 123,
    geometry_features: int = 8,
    rff_features: int = 16,
    reward_hidden_dims: Sequence[int] = (64, 64),
    reward_max_steps: int = 200,
    reward_patience: int = 15,
    reward_feature_cap: int = 32,
    fqe_iterations: int = 120,
    fqe_patience: int = 10,
    value_strata: int = 4,
    max_group_weight: float = 0.25,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Score held-out moment balance with learned reward/value features.

    The reward representation and FQE value feature are fitted on the training
    folds only. The grouped score keeps one large learned feature block from
    swamping mass/reward/value moments while still letting MLP features help.
    """

    valid_idx = np.asarray(valid_idx, dtype=np.int64).reshape(-1)
    train_idx = np.asarray(fit.train_idx, dtype=np.int64).reshape(-1)
    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_next = _as_2d(dataset.next_states, "next_states")
    A_next_pi = _as_2d(dataset.next_target_actions, "next_target_actions")
    init_idx = _initial_validation_indices(dataset, valid_idx, S.shape[0])
    S0 = _as_2d(dataset.initial_states, "initial_states")[init_idx]
    A0 = _as_2d(dataset.initial_actions, "initial_actions")[init_idx]
    weights = np.asarray(fit.model.predict_state_action_ratio(S[valid_idx], A[valid_idx], clip=True), dtype=np.float64)

    groups: dict[str, dict[str, Array]] = {
        "mass": {
            "eval": np.ones((valid_idx.size, 1), dtype=np.float64),
            "next": np.ones((valid_idx.size, 1), dtype=np.float64),
            "initial": np.ones((S0.shape[0], 1), dtype=np.float64),
        },
    }
    geometry = _geometry_feature_block(
        train_states=S[train_idx],
        train_actions=A[train_idx],
        eval_states=S[valid_idx],
        eval_actions=A[valid_idx],
        next_states=S_next[valid_idx],
        next_actions=A_next_pi[valid_idx],
        initial_states=S0,
        initial_actions=A0,
        feature_cap=int(geometry_features),
    )
    if geometry["eval"].shape[1]:
        groups["geometry"] = geometry

    if int(rff_features) > 0:
        rff = _rff_feature_block(
            train_states=S[train_idx],
            train_actions=A[train_idx],
            eval_states=S[valid_idx],
            eval_actions=A[valid_idx],
            next_states=S_next[valid_idx],
            next_actions=A_next_pi[valid_idx],
            initial_states=S0,
            initial_actions=A0,
            seed=int(seed) + 44_771 * (fit.fold + 1),
            feature_count=int(rff_features),
        )
        groups["rff"] = rff

    reward_block = _crossfit_reward_feature_block(
        dataset=dataset,
        train_idx=train_idx,
        valid_idx=valid_idx,
        initial_idx=init_idx,
        seed=int(seed) + 17_077 * (fit.fold + 1),
        hidden_dims=tuple(int(width) for width in reward_hidden_dims),
        max_steps=int(reward_max_steps),
        patience=int(reward_patience),
        feature_cap=int(reward_feature_cap),
        include_phi=True,
    )
    if reward_block is not None:
        reward_features = {
            "eval": np.asarray(reward_block["eval"], dtype=np.float64),
            "next": np.asarray(reward_block["next"], dtype=np.float64),
            "initial": np.asarray(reward_block["initial"], dtype=np.float64),
        }
        groups["reward_mlp"] = reward_features
        groups["reward_scalar"] = {key: value[:, :1] for key, value in reward_features.items()}

    value_block = _crossfit_fqe_value_block(
        dataset=dataset,
        train_idx=train_idx,
        valid_idx=valid_idx,
        initial_idx=init_idx,
        seed=int(seed) + 23_111 * (fit.fold + 1),
        num_iterations=int(fqe_iterations),
        patience=int(fqe_patience),
    )
    if value_block is not None:
        q_group = {
            "eval": np.asarray(value_block["eval"], dtype=np.float64),
            "next": np.asarray(value_block["next"], dtype=np.float64),
            "initial": np.asarray(value_block["initial"], dtype=np.float64),
        }
        groups["value_q"] = q_group
        strata = _strata_indicator_block(
            train_scalar=np.asarray(value_block["train"], dtype=np.float64).reshape(-1),
            eval_scalar=q_group["eval"].reshape(-1),
            next_scalar=q_group["next"].reshape(-1),
            initial_scalar=q_group["initial"].reshape(-1),
            bins=int(value_strata),
        )
        if strata["eval"].shape[1]:
            groups["value_strata"] = strata

    group_scores = {
        name: _moment_balance_for_arrays(weights, float(dataset.gamma), block, eps=float(eps))
        for name, block in groups.items()
    }
    finite_scores = np.asarray([score for score in group_scores.values() if np.isfinite(score)], dtype=np.float64)
    if finite_scores.size:
        score = float(np.mean(finite_scores) + float(max_group_weight) * np.max(finite_scores))
        max_group = float(np.max(finite_scores))
    else:
        score = float("nan")
        max_group = float("nan")
    out: dict[str, Any] = {
        "mb_value_grouped": score,
        "mb_value_grouped_groups": int(finite_scores.size),
        "mb_value_grouped_max_group": max_group,
        "mb_value_grouped_reward_features": int(0 if reward_block is None else np.asarray(reward_block["eval"]).shape[1]),
        "mb_value_grouped_value_features": int(0 if value_block is None else np.asarray(value_block["eval"]).shape[1]),
        "mb_value_grouped_available": bool(reward_block is not None or value_block is not None),
    }
    for name, score_value in group_scores.items():
        out[f"mb_value_group_{name}"] = float(score_value)
        out[f"mb_value_group_{name}_features"] = int(np.asarray(groups[name]["eval"]).shape[1])
    return out


def compute_weight_diagnostics(weights: Array, *, raw_weights: Array | None = None) -> dict[str, Any]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(w)
    wf = w[finite]
    n = int(w.shape[0])
    if wf.size == 0:
        return {
            "ess": 0.0,
            "ess_fraction": 0.0,
            "mean_ratio": float("nan"),
            "q95_ratio": float("nan"),
            "q99_ratio": float("nan"),
            "max_ratio": float("nan"),
            "cv_ratio": float("nan"),
            "zero_fraction": float("nan"),
            "nonfinite_fraction": 1.0,
            "clipping_fraction": float("nan"),
            "invalid": True,
        }
    mean = float(np.mean(wf))
    raw = w if raw_weights is None else np.asarray(raw_weights, dtype=np.float64).reshape(-1)
    rawf = raw[np.isfinite(raw)]
    max_ratio = float(np.max(wf))
    clipped = 0.0
    if rawf.size and np.isfinite(max_ratio):
        clipped = float(np.mean(rawf > max_ratio + 1e-10))
    invalid = bool((not np.isfinite(mean)) or mean <= 1e-6 or mean >= 1e3 or np.mean(~finite) > 0.0)
    return {
        "ess": float(_ess(wf)),
        "ess_fraction": float(_ess(wf) / max(n, 1)),
        "mean_ratio": mean,
        "q95_ratio": float(np.quantile(wf, 0.95)),
        "q99_ratio": float(np.quantile(wf, 0.99)),
        "max_ratio": max_ratio,
        "cv_ratio": float(np.std(wf) / (abs(mean) + 1e-12)),
        "zero_fraction": float(np.mean(wf <= 0.0)),
        "nonfinite_fraction": float(np.mean(~finite)),
        "clipping_fraction": clipped,
        "invalid": invalid,
    }


def summarize_cv_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(dict(row))
    summary = []
    for name, group in grouped.items():
        out: dict[str, Any] = {"candidate": name, "folds": int(len(group))}
        for key in (
            "mb",
            "fp",
            "ess_fraction",
            "mean_ratio",
            "q95_ratio",
            "q99_ratio",
            "max_ratio",
            "clipping_fraction",
            "mb_features",
            "mb_reward_features",
            "observed_reward_weighted_mean",
        ):
            values = _finite_values(row.get(key) for row in group)
            out[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
            out[f"{key}_se"] = _se(values)
        out["ess_fraction_min"] = float(np.min(_finite_values(row.get("ess_fraction") for row in group)))
        out["runtime_sec_mean"] = float(np.mean(_finite_values(row.get("runtime_sec") for row in group)))
        out["invalid"] = bool(any(bool(row.get("invalid", False)) for row in group))
        out["stabilization_strength"] = float(np.mean(_finite_values(row.get("stabilization_strength") for row in group)))
        for key in ("fixed_point_damping", "occupancy_ratio_max", "shrinkage", "ope_value_estimate", "ope_value_abs_error"):
            values = _finite_values(row.get(key) for row in group)
            out[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
        for key in ("oracle_l1", "oracle_l2", "oracle_corr", "oracle_ope_abs_error"):
            values = _finite_values(row.get(key) for row in group)
            out[f"{key}_mean"] = float(np.mean(values)) if values.size else float("nan")
        summary.append(out)
    selected = _select_candidate(summary)
    for row in summary:
        row["selected_by_mb"] = bool(selected is not None and row["candidate"] == selected)
    return sorted(summary, key=lambda row: (not row["selected_by_mb"], row.get("mb_mean", float("inf"))))


def plot_cv_diagnostics(
    rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
) -> list[Path]:
    """Write compact diagnostic plots and return their paths."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(name: str) -> None:
        path = output / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    s = list(summary)
    if s:
        plt.figure(figsize=(5.0, 3.2))
        plt.scatter([r["stabilization_strength"] for r in s], [r["mb_mean"] for r in s])
        for r in s:
            plt.annotate(str(r["candidate"]), (r["stabilization_strength"], r["mb_mean"]), fontsize=7)
        plt.xlabel("stabilization strength")
        plt.ylabel("MB")
        save("mb_vs_stabilization.png")

        plt.figure(figsize=(5.0, 3.2))
        plt.scatter([r["fp_mean"] for r in s], [r["mb_mean"] for r in s])
        for r in s:
            plt.annotate(str(r["candidate"]), (r["fp_mean"], r["mb_mean"]), fontsize=7)
        plt.xlabel("FP")
        plt.ylabel("MB")
        save("fp_vs_mb.png")

        plt.figure(figsize=(5.0, 3.2))
        plt.scatter([r["ess_fraction_mean"] for r in s], [r["mb_mean"] for r in s])
        for r in s:
            plt.annotate(str(r["candidate"]), (r["ess_fraction_mean"], r["mb_mean"]), fontsize=7)
        plt.xlabel("ESS/n")
        plt.ylabel("MB")
        save("ess_vs_mb.png")

    by_candidate: dict[str, list[float]] = {}
    truth = []
    estimated = []
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).extend(row.get("weights", []))
        if row.get("true_ratio") is not None:
            truth.extend(row.get("true_ratio", []))
            estimated.extend(row.get("weights", []))
    if by_candidate:
        plt.figure(figsize=(5.4, 3.4))
        for name, values in by_candidate.items():
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                plt.hist(np.clip(arr, 0.0, np.quantile(arr, 0.99)), bins=30, alpha=0.35, density=True, label=name)
        plt.xlabel("estimated ratio")
        plt.ylabel("density")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, fontsize=7)
        save("estimated_ratio_histogram.png")

    if truth and estimated:
        x = np.asarray(truth, dtype=np.float64)
        y = np.asarray(estimated, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            plt.figure(figsize=(4.2, 4.0))
            plt.scatter(x[mask], y[mask], s=8, alpha=0.45)
            lim = float(np.nanmax([np.max(x[mask]), np.max(y[mask])]))
            plt.plot([0.0, lim], [0.0, lim], color="black", linewidth=1.0)
            plt.xlabel("true ratio")
            plt.ylabel("estimated ratio")
            save("true_vs_estimated_ratio.png")

    return paths


def run_fori_cv_benchmark(
    dataset: BenchmarkDataset,
    candidates: Sequence[FORICVCandidate],
    *,
    k_folds: int = 3,
    seed: int = 123,
    rff_features: int = 16,
    output_dir: str | Path | None = None,
    keep_fold_weights: bool = True,
) -> FORICVResult:
    folds = _make_folds(dataset.n, int(k_folds), int(seed))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for fold, valid_idx in enumerate(folds):
            train_idx = _complement_indices(dataset.n, valid_idx)
            fit = fit_fori_cv_candidate(dataset, candidate, train_idx, fold=fold, seed=seed)
            mb = score_moment_balance(fit, dataset, valid_idx, seed=seed, rff_features=rff_features)
            fp = score_fixed_point_residual(fit, dataset, valid_idx, seed=seed)
            weights = np.asarray(
                fit.model.predict_state_action_ratio(dataset.states[valid_idx], dataset.actions[valid_idx], clip=True),
                dtype=np.float64,
            )
            raw = np.asarray(
                fit.model.predict_state_action_ratio(dataset.states[valid_idx], dataset.actions[valid_idx], clip=False),
                dtype=np.float64,
            )
            row = {
                "candidate": str(candidate.name),
                "fold": int(fold),
                "runtime_sec": float(fit.runtime_sec),
                "stabilization_strength": _stabilization_strength(candidate, fit.model),
                **mb,
                **fp,
                **compute_weight_diagnostics(weights, raw_weights=raw),
                **_stabilization_settings(candidate, fit.model),
                **_fold_value_and_oracle_metrics(dataset, valid_idx, weights),
            }
            if keep_fold_weights:
                row["weights"] = [float(x) for x in weights]
                row["true_ratio"] = None if dataset.true_ratio is None else [
                    float(x) for x in np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1)[valid_idx]
                ]
            rows.append(row)
    summary = summarize_cv_results(rows)
    selected = next((str(row["candidate"]) for row in summary if row.get("selected_by_mb")), None)
    plots = [] if output_dir is None else plot_cv_diagnostics(rows, summary, output_dir=output_dir)
    return FORICVResult(rows=rows, summary=summary, selected_candidate=selected, plot_paths=plots)


def _forward_adjoint_update(fit: FORICVFit, dataset: BenchmarkDataset, valid_idx: Array, w_train: Array, *, seed: int) -> Array:
    model = fit.model
    legacy = model.to_legacy_dict()
    train_idx = np.asarray(fit.train_idx, dtype=np.int64)
    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_val = S[valid_idx]
    A_val = A[valid_idx]
    X_val = np.concatenate([S_val, A_val], axis=1)
    source_state = _predict_source_state_ratio(model, S_val)
    source_weight = _predict_source_weight(model, S_val, A_val)
    if str(fit.candidate.family) == "boosted":
        builder = make_forward_occupancy_dataset(
            bst_k=legacy["bst_k"],
            bst_iw=legacy["bst_iw"],
            k_prediction_offset=float(legacy.get("k_prediction_offset", 0.0)),
            iw_prediction_offset=float(legacy.get("iw_prediction_offset", 0.0)),
            X_sa_kernel=np.concatenate([S[train_idx], A[train_idx]], axis=1),
            X_s_query=S_val,
            X_sa_iw=np.concatenate([S[train_idx], A[train_idx]], axis=1),
            X_sa_query_iw=X_val,
            gamma=float(dataset.gamma),
            mcmc_samples=max(8, int(legacy.get("mcmc_samples") or 24)),
            seed=int(seed) + 3_577 * (fit.fold + 1),
            clip_w_query_max=legacy.get("iw_prediction_max"),
            clip_k_max=legacy.get("k_prediction_max"),
            source_state_ratio_query=None if source_state is None else source_state,
            w_source_query=source_weight if source_state is None else None,
        )
        return np.asarray(builder(w_beh=w_train, w_old_query=model.predict_state_action_ratio(S_val, A_val, clip=True))["y"], dtype=np.float64)
    config = fit.candidate.occupancy if isinstance(fit.candidate.occupancy, NeuralOccupancyRegressionConfig) else NeuralOccupancyRegressionConfig()
    builder = _NeuralTargetBuilder(
        X_sa_kernel=np.concatenate([S[train_idx], A[train_idx]], axis=1),
        X_s_query=S_val,
        X_sa_query_iw=X_val,
        gamma=float(dataset.gamma),
        mcmc_samples=max(8, int(config.mcmc_samples)),
        batch_query=max(1, int(config.batch_size)),
        action_predictor=model.action_ratio_predictor,
        transition_predictor=model.transition_ratio_predictor,
        seed=int(seed) + 3_577 * (fit.fold + 1),
        normalize_transition_cache=bool(getattr(config, "normalize_transition_cache", False)),
        transition_cache_norm_eps=float(getattr(config, "transition_cache_norm_eps", 1e-12)),
        source_state_ratio_query=None if source_state is None else source_state,
        w_source_query=source_weight if source_state is None else None,
    )
    return np.asarray(builder(w_beh=w_train)["y"], dtype=np.float64)


def _stabilize_update(raw_update: Array, current: Array, legacy: dict[str, Any]) -> Array:
    update, _ = _make_stabilized_fixed_point_target(
        raw_target=np.asarray(raw_update, dtype=np.float64),
        current=np.asarray(current, dtype=np.float64),
        eta=float(legacy.get("fixed_point_damping", 1.0) or 1.0),
        normalize=bool(legacy.get("normalize_occupancy", False)),
        occupancy_ratio_max=legacy.get("occupancy_ratio_max"),
        eps=float(legacy.get("occupancy_projection_eps", 1e-12) or 1e-12),
        clip_pseudo_outcomes=bool(legacy.get("clip_pseudo_outcomes", False)),
        pseudo_outcome_max=legacy.get("pseudo_outcome_max"),
        pseudo_outcome_upper_quantile=float(legacy.get("pseudo_outcome_upper_quantile", 0.995) or 0.995),
        pseudo_outcome_min=0.0,
        target_min=0.0,
        target_max=legacy.get("occupancy_ratio_max"),
    )
    return np.asarray(update, dtype=np.float64)


def _predict_source_weight(model: Any, states: Array, actions: Array) -> Array:
    legacy = model.to_legacy_dict()
    X = np.concatenate([_as_2d(states, "states"), _as_2d(actions, "actions")], axis=1)
    if bool(legacy.get("initial_joint_ratio_enabled", False)):
        return _predict_source_fit(legacy.get("source_fit"), X)
    action = np.asarray(model.predict_action_ratio(states, actions, clip=True), dtype=np.float64).reshape(-1)
    source_state = _predict_source_state_ratio(model, states)
    if source_state is None:
        return action
    return np.maximum(action * source_state, 0.0)


def _predict_source_state_ratio(model: Any, states: Array) -> Array | None:
    legacy = model.to_legacy_dict()
    if not bool(legacy.get("source_state_ratio_enabled", False)):
        return None
    return _predict_source_fit(legacy.get("source_fit"), _as_2d(states, "states"))


def _predict_one_step_ratio(model: Any, x_sa: Array) -> Array | None:
    legacy = model.to_legacy_dict()
    if not bool(legacy.get("one_step_direct_ratio_enabled", False)):
        return None
    return _predict_source_fit(legacy.get("c_fit"), x_sa)


def _predict_source_fit(fit: Any, x: Array) -> Array:
    if not isinstance(fit, dict):
        return np.ones(np.asarray(x).shape[0], dtype=np.float64)
    if "predictor" in fit:
        return np.asarray(fit["predictor"].predict(x, postprocess=True), dtype=np.float64).reshape(-1)
    if "bst_source" in fit:
        return np.asarray(_predict_processed_source_state_ratio(fit=fit, X=x), dtype=np.float64).reshape(-1)
    return np.ones(np.asarray(x).shape[0], dtype=np.float64)


def _moment_features(
    *,
    train_states: Array,
    train_actions: Array,
    eval_states: Array,
    eval_actions: Array,
    next_states: Array,
    next_actions: Array,
    initial_states: Array,
    initial_actions: Array,
    seed: int,
    rff_features: int,
    raw_features: bool = True,
    reward_block: dict[str, Array] | None = None,
) -> dict[str, Array]:
    z_train = np.concatenate([_as_2d(train_states, "train_states"), _as_2d(train_actions, "train_actions")], axis=1)
    mean = np.mean(z_train, axis=0)
    scale = np.std(z_train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    def std(states: Array, actions: Array) -> Array:
        z = np.concatenate([_as_2d(states, "states"), _as_2d(actions, "actions")], axis=1)
        return (z - mean.reshape(1, -1)) / scale.reshape(1, -1)

    z_eval = std(eval_states, eval_actions)
    z_next = std(next_states, next_actions)
    z_initial = std(initial_states, initial_actions)
    rng = np.random.default_rng(int(seed))
    rff_count = max(0, int(rff_features))
    if rff_count:
        W = rng.normal(size=(z_train.shape[1], rff_count)) / np.sqrt(max(z_train.shape[1], 1))
        b = rng.uniform(0.0, 2.0 * np.pi, size=rff_count)

        def rff(z: Array) -> Array:
            return np.sqrt(2.0 / max(rff_count, 1)) * np.cos(z @ W + b.reshape(1, -1))
    else:
        def rff(z: Array) -> Array:
            return np.empty((z.shape[0], 0), dtype=np.float64)

    def assemble(z: Array, reward_key: str) -> Array:
        blocks = [np.ones((z.shape[0], 1), dtype=np.float64)]
        if bool(raw_features):
            blocks.append(z)
        blocks.append(rff(z))
        if reward_block is not None:
            blocks.append(np.asarray(reward_block[reward_key], dtype=np.float64))
        return np.concatenate(blocks, axis=1)

    reward_count = 0 if reward_block is None else int(np.asarray(reward_block["eval"], dtype=np.float64).shape[1])
    return {
        "eval": assemble(z_eval, "eval"),
        "next": assemble(z_next, "next"),
        "initial": assemble(z_initial, "initial"),
        "reward_feature_count": reward_count,
    }


def _moment_balance_for_arrays(weights: Array, gamma: float, block: dict[str, Array], *, eps: float) -> float:
    eval_features = np.asarray(block["eval"], dtype=np.float64)
    next_features = np.asarray(block["next"], dtype=np.float64)
    initial_features = np.asarray(block["initial"], dtype=np.float64)
    if eval_features.ndim != 2 or next_features.ndim != 2 or initial_features.ndim != 2 or eval_features.shape[1] == 0:
        return float("nan")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid_term = w[:, None] * (eval_features - float(gamma) * next_features)
    initial_term = (1.0 - float(gamma)) * initial_features
    deltas = np.mean(valid_term, axis=0) - np.mean(initial_term, axis=0)
    var_valid = np.var(valid_term, axis=0, ddof=1) / max(valid_term.shape[0], 1)
    var_initial = np.var(initial_term, axis=0, ddof=1) / max(initial_term.shape[0], 1)
    sigma2 = np.maximum(var_valid + var_initial, float(eps))
    z2 = deltas**2 / sigma2
    z2 = z2[np.isfinite(z2)]
    return float(np.mean(z2)) if z2.size else float("nan")


def _geometry_feature_block(
    *,
    train_states: Array,
    train_actions: Array,
    eval_states: Array,
    eval_actions: Array,
    next_states: Array,
    next_actions: Array,
    initial_states: Array,
    initial_actions: Array,
    feature_cap: int,
) -> dict[str, Array]:
    train = np.concatenate([_as_2d(train_states, "train_states"), _as_2d(train_actions, "train_actions")], axis=1)
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    def std(states: Array, actions: Array) -> Array:
        z = np.concatenate([_as_2d(states, "states"), _as_2d(actions, "actions")], axis=1)
        return (z - mean.reshape(1, -1)) / scale.reshape(1, -1)

    train_z = std(train_states, train_actions)
    cap = max(0, min(int(feature_cap), train_z.shape[1]))
    if cap == 0:
        empty = np.empty((np.asarray(eval_states).shape[0], 0), dtype=np.float64)
        return {
            "eval": empty,
            "next": np.empty((np.asarray(next_states).shape[0], 0), dtype=np.float64),
            "initial": np.empty((np.asarray(initial_states).shape[0], 0), dtype=np.float64),
        }
    if cap < train_z.shape[1]:
        _, _, vt = np.linalg.svd(train_z - np.mean(train_z, axis=0, keepdims=True), full_matrices=False)
        components = vt[:cap].T

        def transform(states: Array, actions: Array) -> Array:
            return std(states, actions) @ components
    else:
        def transform(states: Array, actions: Array) -> Array:
            return std(states, actions)

    return {
        "eval": transform(eval_states, eval_actions),
        "next": transform(next_states, next_actions),
        "initial": transform(initial_states, initial_actions),
    }


def _rff_feature_block(
    *,
    train_states: Array,
    train_actions: Array,
    eval_states: Array,
    eval_actions: Array,
    next_states: Array,
    next_actions: Array,
    initial_states: Array,
    initial_actions: Array,
    seed: int,
    feature_count: int,
) -> dict[str, Array]:
    base = _geometry_feature_block(
        train_states=train_states,
        train_actions=train_actions,
        eval_states=eval_states,
        eval_actions=eval_actions,
        next_states=next_states,
        next_actions=next_actions,
        initial_states=initial_states,
        initial_actions=initial_actions,
        feature_cap=10_000,
    )
    z_train = np.concatenate([_as_2d(train_states, "train_states"), _as_2d(train_actions, "train_actions")], axis=1)
    dim = int(z_train.shape[1])
    count = max(0, int(feature_count))
    if count == 0:
        return {
            "eval": np.empty((base["eval"].shape[0], 0), dtype=np.float64),
            "next": np.empty((base["next"].shape[0], 0), dtype=np.float64),
            "initial": np.empty((base["initial"].shape[0], 0), dtype=np.float64),
        }
    rng = np.random.default_rng(int(seed))
    W = rng.normal(size=(dim, count)) / np.sqrt(max(dim, 1))
    b = rng.uniform(0.0, 2.0 * np.pi, size=count)

    def transform(z: Array) -> Array:
        return np.sqrt(2.0 / max(count, 1)) * np.cos(np.asarray(z, dtype=np.float64) @ W + b.reshape(1, -1))

    return {key: transform(value) for key, value in base.items()}


def _strata_indicator_block(
    *,
    train_scalar: Array,
    eval_scalar: Array,
    next_scalar: Array,
    initial_scalar: Array,
    bins: int,
) -> dict[str, Array]:
    train = np.asarray(train_scalar, dtype=np.float64).reshape(-1)
    train = train[np.isfinite(train)]
    bin_count = max(0, int(bins))
    if train.size < 8 or bin_count <= 1:
        return {
            "eval": np.empty((np.asarray(eval_scalar).shape[0], 0), dtype=np.float64),
            "next": np.empty((np.asarray(next_scalar).shape[0], 0), dtype=np.float64),
            "initial": np.empty((np.asarray(initial_scalar).shape[0], 0), dtype=np.float64),
        }
    probs = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    thresholds = np.unique(np.quantile(train, probs))
    if thresholds.size == 0:
        return {
            "eval": np.empty((np.asarray(eval_scalar).shape[0], 0), dtype=np.float64),
            "next": np.empty((np.asarray(next_scalar).shape[0], 0), dtype=np.float64),
            "initial": np.empty((np.asarray(initial_scalar).shape[0], 0), dtype=np.float64),
        }

    train_bins = np.digitize(train, thresholds, right=False)
    levels = int(thresholds.size + 1)
    train_ind = np.eye(levels, dtype=np.float64)[train_bins]
    mean = np.mean(train_ind, axis=0)
    scale = np.std(train_ind, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    def transform(values: Array) -> Array:
        idx = np.digitize(np.asarray(values, dtype=np.float64).reshape(-1), thresholds, right=False)
        ind = np.eye(levels, dtype=np.float64)[idx]
        return (ind - mean.reshape(1, -1)) / scale.reshape(1, -1)

    return {
        "eval": transform(eval_scalar),
        "next": transform(next_scalar),
        "initial": transform(initial_scalar),
    }


def _crossfit_fqe_value_block(
    *,
    dataset: BenchmarkDataset,
    train_idx: Array,
    valid_idx: Array,
    initial_idx: Array,
    seed: int,
    num_iterations: int,
    patience: int,
) -> dict[str, Array] | None:
    try:
        from fqe import BoostedFQEConfig, fit_fqe_lgbm
    except Exception:
        return None

    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_next = _as_2d(dataset.next_states, "next_states")
    A_next = _as_2d(dataset.next_target_actions, "next_target_actions")
    S0 = _as_2d(dataset.initial_states, "initial_states")
    A0 = _as_2d(dataset.initial_actions, "initial_actions")
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    if train_idx.size < 8 or rewards.shape[0] != S.shape[0]:
        return None
    masks = np.asarray(dataset.masks, dtype=np.float64).reshape(-1)
    terminals = np.clip(1.0 - masks[train_idx], 0.0, 1.0) if masks.shape[0] == S.shape[0] else None
    lgb_params = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": max(2, min(20, train_idx.size // 5)),
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": 1,
    }
    config = BoostedFQEConfig.stable_defaults(
        num_iterations=max(1, int(num_iterations)),
        patience=max(1, int(patience)),
        validation_fraction=0.2,
        seed=int(seed),
        refit_on_all_data=True,
        lgb_params=lgb_params,
    )
    try:
        model = fit_fqe_lgbm(
            states=S[train_idx],
            actions=A[train_idx],
            next_states=S_next[train_idx],
            next_actions=A_next[train_idx],
            rewards=rewards[train_idx],
            gamma=float(dataset.gamma),
            terminals=terminals,
            config=config,
        )
    except Exception:
        return None

    q_train = np.asarray(model.predict_q(S[train_idx], A[train_idx]), dtype=np.float64).reshape(-1, 1)
    q_eval = np.asarray(model.predict_q(S[valid_idx], A[valid_idx]), dtype=np.float64).reshape(-1, 1)
    q_next = np.asarray(model.predict_q(S_next[valid_idx], A_next[valid_idx]), dtype=np.float64).reshape(-1, 1)
    q_initial = np.asarray(model.predict_q(S0[initial_idx], A0[initial_idx]), dtype=np.float64).reshape(-1, 1)
    mean = np.mean(q_train, axis=0)
    scale = np.std(q_train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    def std(x: Array) -> Array:
        return (np.asarray(x, dtype=np.float64) - mean.reshape(1, -1)) / scale.reshape(1, -1)

    return {
        "train": std(q_train),
        "eval": std(q_eval),
        "next": std(q_next),
        "initial": std(q_initial),
    }


def _crossfit_reward_feature_block(
    *,
    dataset: BenchmarkDataset,
    train_idx: Array,
    valid_idx: Array,
    initial_idx: Array,
    seed: int,
    hidden_dims: Sequence[int],
    max_steps: int,
    patience: int,
    feature_cap: int,
    include_phi: bool = True,
) -> dict[str, Array] | None:
    try:
        import torch
        from torch import nn
    except Exception:
        return None

    _stabilize_torch_runtime()
    S = _as_2d(dataset.states, "states")
    A = _as_2d(dataset.actions, "actions")
    S_next = _as_2d(dataset.next_states, "next_states")
    A_next = _as_2d(dataset.next_target_actions, "next_target_actions")
    S0 = _as_2d(dataset.initial_states, "initial_states")
    A0 = _as_2d(dataset.initial_actions, "initial_actions")
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    if train_idx.size < 8 or rewards.shape[0] != S.shape[0]:
        return None

    x_train_raw = np.concatenate([S[train_idx], A[train_idx]], axis=1).astype(np.float64)
    x_eval_raw = np.concatenate([S[valid_idx], A[valid_idx]], axis=1).astype(np.float64)
    x_next_raw = np.concatenate([S_next[valid_idx], A_next[valid_idx]], axis=1).astype(np.float64)
    x_initial_raw = np.concatenate([S0[initial_idx], A0[initial_idx]], axis=1).astype(np.float64)
    x_mean = np.mean(x_train_raw, axis=0)
    x_scale = np.std(x_train_raw, axis=0)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)

    def standardize_x(x: Array) -> Array:
        return ((np.asarray(x, dtype=np.float64) - x_mean.reshape(1, -1)) / x_scale.reshape(1, -1)).astype(np.float32)

    y_train = rewards[train_idx].astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_scale = float(np.std(y_train))
    if not np.isfinite(y_scale) or y_scale <= 1e-8:
        y_scale = 1.0
    y_std = ((y_train - y_mean) / y_scale).astype(np.float32)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(train_idx.size)
    valid_count = max(1, int(round(0.2 * train_idx.size)))
    inner_valid = perm[:valid_count]
    inner_train = perm[valid_count:]
    if inner_train.size == 0:
        inner_train = inner_valid
    torch.manual_seed(int(seed))
    device = torch.device("cpu")
    model = _RewardFeatureMLP(
        input_dim=x_train_raw.shape[1],
        hidden_dims=tuple(hidden_dims),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_train = torch.as_tensor(standardize_x(x_train_raw), dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    batch_size = min(256, max(16, inner_train.size))
    for step in range(max(1, int(max_steps))):
        batch_np = rng.choice(inner_train, size=batch_size, replace=inner_train.size < batch_size)
        batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
        pred, _ = model(x_train[batch])
        loss = torch.mean((pred - y_t[batch]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 5 == 0 or step == int(max_steps) - 1:
            model.eval()
            with torch.no_grad():
                inner_idx = torch.as_tensor(inner_valid, dtype=torch.long, device=device)
                valid_pred, _ = model(x_train[inner_idx])
                valid_loss = float(torch.mean((valid_pred - y_t[inner_idx]) ** 2).item())
            model.train()
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= max(1, int(patience)):
                    break
    model.load_state_dict(best_state)
    model.eval()

    def transform(x_raw: Array) -> tuple[Array, Array]:
        with torch.no_grad():
            pred_std, phi = model(torch.as_tensor(standardize_x(x_raw), dtype=torch.float32, device=device))
        pred = pred_std.detach().cpu().numpy().reshape(-1, 1).astype(np.float64) * y_scale + y_mean
        phi_arr = phi.detach().cpu().numpy().astype(np.float64)
        if not bool(include_phi):
            phi_arr = np.empty((pred.shape[0], 0), dtype=np.float64)
        elif int(feature_cap) > 0:
            phi_arr = phi_arr[:, : int(feature_cap)]
        return pred, phi_arr

    pred_train, phi_train = transform(x_train_raw)
    pred_eval, phi_eval = transform(x_eval_raw)
    pred_next, phi_next = transform(x_next_raw)
    pred_initial, phi_initial = transform(x_initial_raw)
    train_features = np.concatenate([pred_train, phi_train], axis=1)
    f_mean = np.mean(train_features, axis=0)
    f_scale = np.std(train_features, axis=0)
    f_scale = np.where(f_scale > 1e-8, f_scale, 1.0)

    def standardize_features(pred: Array, phi: Array) -> Array:
        feats = np.concatenate([pred, phi], axis=1)
        return (feats - f_mean.reshape(1, -1)) / f_scale.reshape(1, -1)

    return {
        "train": standardize_features(pred_train, phi_train),
        "eval": standardize_features(pred_eval, phi_eval),
        "next": standardize_features(pred_next, phi_next),
        "initial": standardize_features(pred_initial, phi_initial),
    }


class _RewardFeatureMLP:
    def __init__(self, *, input_dim: int, hidden_dims: Sequence[int]) -> None:
        import torch
        from torch import nn

        self._nn = nn
        layers = []
        prev = int(input_dim)
        dims = tuple(int(width) for width in hidden_dims) or (64, 64)
        for width in dims:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.SiLU())
            prev = int(width)
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        self._module = nn.Module()
        self._module.body = self.body
        self._module.head = self.head

    def __call__(self, x: Any) -> tuple[Any, Any]:
        phi = self.body(x)
        return self.head(phi).reshape(-1), phi

    def to(self, device: Any) -> "_RewardFeatureMLP":
        self._module.to(device)
        return self

    def parameters(self) -> Any:
        return self._module.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self._module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._module.load_state_dict(state)

    def train(self) -> None:
        self._module.train()

    def eval(self) -> None:
        self._module.eval()


def _observed_reward_diagnostic(dataset: BenchmarkDataset, valid_idx: Array, weights: Array) -> dict[str, float]:
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    if rewards.shape[0] <= np.max(valid_idx, initial=-1):
        return {}
    r = rewards[valid_idx]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if r.shape[0] != w.shape[0]:
        return {}
    return {
        "observed_reward_weighted_mean": float(np.mean(w * r)),
        "observed_reward_unweighted_mean": float(np.mean(r)),
    }


def _fold_value_and_oracle_metrics(dataset: BenchmarkDataset, valid_idx: Array, weights: Array) -> dict[str, Any]:
    rewards = np.asarray(dataset.rewards, dtype=np.float64).reshape(-1)[valid_idx]
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {"ope_value_estimate": float(np.mean(weights * rewards))}
    target = dataset.metadata.get("target_policy_value")
    if target is not None:
        out["ope_value_abs_error"] = float(abs(out["ope_value_estimate"] - float(target)))
    if dataset.true_ratio is not None:
        truth = np.asarray(dataset.true_ratio, dtype=np.float64).reshape(-1)[valid_idx]
        diff = weights - truth
        out["oracle_l1"] = float(np.mean(np.abs(diff)))
        out["oracle_l2"] = float(np.sqrt(np.mean(diff**2)))
        if np.std(weights) > 1e-12 and np.std(truth) > 1e-12:
            out["oracle_corr"] = float(np.corrcoef(weights, truth)[0, 1])
        out["oracle_ope_abs_error"] = float(abs(np.mean(weights * rewards) - np.mean(truth * rewards)))
    return out


def _stabilization_settings(candidate: FORICVCandidate, model: Any) -> dict[str, Any]:
    legacy = model.to_legacy_dict()
    return {
        "fixed_point_damping": _float_or_nan(legacy.get("fixed_point_damping")),
        "occupancy_ratio_max": _float_or_nan(legacy.get("occupancy_ratio_max")),
        "shrinkage": _float_or_nan(candidate.metadata.get("shrinkage", np.nan)),
        "candidate_family": str(candidate.family),
    }


def _stabilization_strength(candidate: FORICVCandidate, model: Any) -> float:
    legacy = model.to_legacy_dict()
    damping = _float_or_nan(legacy.get("fixed_point_damping"))
    cap = _float_or_nan(legacy.get("occupancy_ratio_max"))
    strength = 0.0
    if np.isfinite(damping):
        strength += max(0.0, 1.0 - damping)
    if np.isfinite(cap) and cap > 0:
        strength += 1.0 / cap
    if bool(legacy.get("normalize_occupancy", False)):
        strength += 0.05
    shrinkage = _float_or_nan(candidate.metadata.get("shrinkage", np.nan))
    if np.isfinite(shrinkage):
        strength += max(0.0, shrinkage)
    return float(strength)


def _select_candidate(summary: Sequence[dict[str, Any]]) -> str | None:
    valid = [row for row in summary if not bool(row.get("invalid", False)) and np.isfinite(float(row.get("mb_mean", np.inf)))]
    if not valid:
        return None
    best = min(valid, key=lambda row: float(row["mb_mean"]))
    best_mean = float(best["mb_mean"])
    best_se = float(best.get("mb_se", 0.0) or 0.0)
    tied = [
        row
        for row in valid
        if float(row["mb_mean"]) <= best_mean + max(best_se, float(row.get("mb_se", 0.0) or 0.0))
    ]
    return str(min(tied, key=lambda row: (float(row.get("stabilization_strength", 0.0)), float(row.get("mb_mean", np.inf))))["candidate"])


def _initial_validation_indices(dataset: BenchmarkDataset, valid_idx: Array, n: int) -> Array:
    initial = np.asarray(dataset.initial_states)
    if initial.shape[0] == int(n):
        return np.asarray(valid_idx, dtype=np.int64)
    return np.arange(initial.shape[0], dtype=np.int64)


def _make_folds(n: int, k_folds: int, seed: int) -> list[Array]:
    if int(k_folds) < 2:
        raise ValueError("k_folds must be at least 2.")
    rng = np.random.default_rng(int(seed))
    indices = np.arange(int(n), dtype=np.int64)
    rng.shuffle(indices)
    return [fold.astype(np.int64, copy=False) for fold in np.array_split(indices, int(k_folds))]


def _complement_indices(n: int, valid_idx: Array) -> Array:
    mask = np.ones(int(n), dtype=bool)
    mask[np.asarray(valid_idx, dtype=np.int64)] = False
    return np.flatnonzero(mask)


def _finite_values(values: Iterable[Any]) -> Array:
    out = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def _se(values: Array) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _stabilize_torch_runtime() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
