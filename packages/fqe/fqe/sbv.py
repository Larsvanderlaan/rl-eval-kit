from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from fqe.fit_fqe import Array, _as_1d_float, _as_2d_float, _optional_terminals, _validate_gamma


try:  # pragma: no cover - exercised when torch is installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent.
    torch = None
    nn = None


__all__ = [
    "TransitionDataset",
    "FQECandidate",
    "SBVSelectionResult",
    "split_by_episode_ids",
    "expected_q_under_policy",
    "estimate_policy_value_from_q",
    "compute_candidate_logged_q_matrix",
    "compute_candidate_next_value_matrix",
    "select_td_with_sbv_audit",
    "LowRankOperatorSBVValidator",
    "GenerativeBellmanValidator",
    "DirectMultiOutputSBVValidator",
]


@dataclass
class TransitionDataset:
    """Offline transitions grouped by trajectory for FQE validation."""

    obs: Array
    actions: Array
    rewards: Array
    next_obs: Array
    done: Array
    episode_id: Array | None = None
    timestep: Array | None = None
    sample_weight: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rewards = _as_1d_float(self.rewards, "rewards")
        obs = _as_2d_float(self.obs, "obs", n_rows=rewards.shape[0])
        next_obs = _as_2d_float(self.next_obs, "next_obs", n_rows=rewards.shape[0])
        actions = _as_2d_action(self.actions, "actions", n_rows=rewards.shape[0])
        done = _optional_terminals(self.done, rewards.shape[0])
        episode_id = np.arange(rewards.shape[0]) if self.episode_id is None else np.asarray(self.episode_id).reshape(-1)
        if episode_id.shape[0] != rewards.shape[0]:
            raise ValueError("episode_id must have one entry per transition.")
        if self.timestep is None:
            timestep = np.zeros(rewards.shape[0], dtype=np.int64)
        else:
            timestep = np.asarray(self.timestep).reshape(-1)
            if timestep.shape[0] != rewards.shape[0]:
                raise ValueError("timestep must have one entry per transition.")
        weight = None
        if self.sample_weight is not None:
            weight = np.asarray(self.sample_weight, dtype=np.float64).reshape(-1)
            if weight.shape[0] != rewards.shape[0]:
                raise ValueError("sample_weight must have one entry per transition.")
            if not np.all(np.isfinite(weight)):
                raise ValueError("sample_weight must contain only finite values.")
            if np.any(weight < 0.0):
                raise ValueError("sample_weight must be nonnegative.")
            if float(np.sum(weight)) <= 0.0:
                raise ValueError("sample_weight must have positive total weight.")
        object.__setattr__(self, "obs", obs)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "next_obs", next_obs)
        object.__setattr__(self, "done", done)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "timestep", timestep)
        object.__setattr__(self, "sample_weight", weight)

    @property
    def n(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def states(self) -> Array:
        return self.obs

    @property
    def next_states(self) -> Array:
        return self.next_obs

    @property
    def terminals(self) -> Array:
        return self.done

    def subset(self, indices: Array | Sequence[int]) -> "TransitionDataset":
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        return TransitionDataset(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            episode_id=self.episode_id[idx],
            timestep=self.timestep[idx],
            sample_weight=None if self.sample_weight is None else self.sample_weight[idx],
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class FQECandidate:
    """A fitted Q/FQE candidate plus selection metadata."""

    candidate_id: str
    model: Any
    checkpoint_path: str | Path | None = None
    fqe_iteration: int | None = None
    hyperparams: Mapping[str, Any] | None = None
    complexity_order_key: Any = None
    trained_on_split_ids: Iterable[Any] | None = None

    def complexity_key(self, index: int) -> tuple[Any, ...]:
        if self.complexity_order_key is not None:
            key = self.complexity_order_key
            return tuple(key) if isinstance(key, tuple) else (key,)
        if self.fqe_iteration is not None:
            return (int(self.fqe_iteration),)
        return (int(index),)


@dataclass
class SBVSelectionResult:
    """Selection table and fitted-validator diagnostics."""

    method: str
    rows: list[dict[str, Any]]
    selected_candidate_id: str
    selected_one_se_candidate_id: str | None
    selected_index: int
    selected_one_se_index: int | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def selected_row(self) -> dict[str, Any]:
        return dict(self.rows[self.selected_index])


def select_td_with_sbv_audit(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[FQECandidate | Any] | None = None,
    *,
    td_score_key: str = "naive_td_score",
    sbv_score_key: str = "sbv_score",
    method: str = "td_sbv_audit",
    use_one_se: bool = True,
    strong_sbv_relative_margin: float = 0.05,
) -> SBVSelectionResult:
    """Select by held-out TD and report SBV as a non-vetoing audit signal.

    This is the conservative production rule suggested by the Gym NN-size
    evidence: TD chooses; SBV audits. The input rows are usually
    `LowRankOperatorSBVValidator.score(...).rows`, which contain both
    `sbv_score` and `naive_td_score`.
    """

    if not rows:
        raise ValueError("rows must be nonempty.")
    audit_rows = [dict(row) for row in rows]
    if candidates is None:
        candidate_list = [
            FQECandidate(
                candidate_id=str(row.get("candidate_id", f"candidate_{idx:03d}")),
                model=None,
                fqe_iteration=row.get("fqe_iteration"),
                complexity_order_key=idx,
            )
            for idx, row in enumerate(audit_rows)
        ]
    else:
        candidate_list = _as_candidates(candidates)
    if len(candidate_list) != len(audit_rows):
        raise ValueError("candidates and rows must have the same length.")
    for row in audit_rows:
        if td_score_key not in row:
            raise KeyError(f"row is missing {td_score_key!r}.")
        if sbv_score_key not in row:
            raise KeyError(f"row is missing {sbv_score_key!r}.")
        row["sbv_selected_min_score"] = bool(row.get("selected_min_score", False))
        row["sbv_selected_one_se"] = bool(row.get("selected_one_se", False))
        row["selected_min_score"] = False
        row["selected_one_se"] = False
        row["td_selected_min_score"] = False
        row["td_selected_one_se"] = False
    td_min, td_one_se = _mark_selection(audit_rows, candidate_list, td_score_key)
    selected = int(td_one_se if use_one_se else td_min)
    for idx, row in enumerate(audit_rows):
        row["td_selected_min_score"] = bool(idx == td_min)
        row["td_selected_one_se"] = bool(idx == td_one_se)
        row["selected_by_td_sbv_audit"] = bool(idx == selected)
    sbv_scores = np.asarray([float(row[sbv_score_key]) for row in audit_rows], dtype=np.float64)
    td_scores = np.asarray([float(row[td_score_key]) for row in audit_rows], dtype=np.float64)
    sbv_best = int(np.nanargmin(sbv_scores))
    sbv_best_score = float(sbv_scores[sbv_best])
    selected_sbv_score = float(sbv_scores[selected])
    sbv_se_key = f"{sbv_score_key}_se" if f"{sbv_score_key}_se" in audit_rows[sbv_best] else sbv_score_key.replace("score", "score_se")
    sbv_best_se = float(audit_rows[sbv_best].get(sbv_se_key, 0.0))
    margin = max(sbv_best_se, float(strong_sbv_relative_margin) * max(abs(selected_sbv_score), 1.0), 0.0)
    disagree = bool(sbv_best != selected)
    strong_disagree = bool(disagree and sbv_best_score + margin < selected_sbv_score)
    status = "green" if not disagree else "red" if strong_disagree else "yellow"
    for row in audit_rows:
        row["td_sbv_audit_status"] = status
        row["td_sbv_audit_disagreement"] = disagree
        row["td_sbv_audit_strong_disagreement"] = strong_disagree
        row["td_sbv_audit_td_candidate_id"] = candidate_list[selected].candidate_id
        row["td_sbv_audit_sbv_candidate_id"] = candidate_list[sbv_best].candidate_id
        row["td_sbv_audit_recommendation"] = "select_td"
    diagnostics = {
        "selector": "td_primary_sbv_audit",
        "td_score_key": td_score_key,
        "sbv_score_key": sbv_score_key,
        "td_min_index": int(td_min),
        "td_one_se_index": int(td_one_se),
        "selected_by_td_index": int(selected),
        "sbv_min_index": int(sbv_best),
        "td_sbv_audit_td_candidate_id": candidate_list[selected].candidate_id,
        "td_sbv_audit_sbv_candidate_id": candidate_list[sbv_best].candidate_id,
        "td_sbv_audit_status": status,
        "td_sbv_audit_disagreement": disagree,
        "td_sbv_audit_strong_disagreement": strong_disagree,
        "td_selected_score": float(td_scores[selected]),
        "sbv_score_for_td_selection": selected_sbv_score,
        "sbv_best_score": sbv_best_score,
    }
    return SBVSelectionResult(
        method=method,
        rows=audit_rows,
        selected_candidate_id=candidate_list[selected].candidate_id,
        selected_one_se_candidate_id=candidate_list[td_one_se].candidate_id,
        selected_index=selected,
        selected_one_se_index=int(td_one_se),
        diagnostics=diagnostics,
    )


def split_by_episode_ids(
    dataset: TransitionDataset,
    fractions: Mapping[str, float] | Sequence[float],
    seed: int,
) -> dict[str, TransitionDataset]:
    """Split transitions by whole episode ids only."""

    if isinstance(fractions, Mapping):
        names = [str(name) for name in fractions]
        values = np.asarray([float(fractions[name]) for name in fractions], dtype=np.float64)
    else:
        values = np.asarray([float(value) for value in fractions], dtype=np.float64)
        default_names = ("D_Q", "D_B", "D_score", "split_3", "split_4")
        names = [default_names[idx] if idx < len(default_names) else f"split_{idx}" for idx in range(values.shape[0])]
    if values.ndim != 1 or values.shape[0] == 0:
        raise ValueError("fractions must be a nonempty mapping or sequence.")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("fractions must be finite and nonnegative.")
    if float(np.sum(values)) <= 0.0:
        raise ValueError("fractions must have positive total mass.")
    probs = values / float(np.sum(values))
    unique_episode_ids = np.asarray(np.unique(dataset.episode_id))
    rng = np.random.default_rng(int(seed))
    shuffled = unique_episode_ids[rng.permutation(unique_episode_ids.shape[0])]
    cuts = np.floor(np.cumsum(probs[:-1]) * shuffled.shape[0]).astype(np.int64)
    episode_blocks = np.split(shuffled, cuts)
    out: dict[str, TransitionDataset] = {}
    for name, episode_block in zip(names, episode_blocks):
        mask = np.isin(dataset.episode_id, episode_block)
        out[name] = dataset.subset(np.flatnonzero(mask))
    return out


def expected_q_under_policy(
    q_model: Any,
    obs_batch: Array,
    target_policy: Any,
    action_space: Any,
    n_action_samples: int = 1,
    seed: int | None = None,
) -> Array:
    """Compute or Monte Carlo estimate ``E_{a~pi(.|s)} Q(s, a)``."""

    obs = _as_2d_float(obs_batch, "obs_batch")
    discrete_actions = _discrete_action_values(action_space)
    if discrete_actions is not None:
        probs = _policy_action_probabilities(target_policy, obs, discrete_actions.shape[0])
        all_action_q = _predict_all_actions_if_available(q_model, obs, discrete_actions.shape[0])
        if all_action_q is not None:
            return np.sum(probs * all_action_q, axis=1).astype(np.float64)
        q_cols = []
        for action_idx in range(discrete_actions.shape[0]):
            actions = np.repeat(discrete_actions[action_idx : action_idx + 1], obs.shape[0], axis=0)
            q_cols.append(_predict_q(q_model, obs, actions))
        return np.sum(probs * np.column_stack(q_cols), axis=1).astype(np.float64)

    actions = _deterministic_policy_actions(target_policy, obs)
    if actions is not None:
        return _predict_q(q_model, obs, actions)

    if int(n_action_samples) <= 0:
        raise ValueError("n_action_samples must be positive.")
    rng = np.random.default_rng(seed)
    q_samples = []
    for sample_idx in range(int(n_action_samples)):
        sampled = _sample_policy_actions(target_policy, obs, rng, sample_idx=sample_idx, n_samples=int(n_action_samples))
        q_samples.append(_predict_q(q_model, obs, sampled))
    return np.mean(np.stack(q_samples, axis=1), axis=1).astype(np.float64)


def estimate_policy_value_from_q(
    q_model: Any,
    initial_states: Array,
    target_policy: Any,
    action_space: Any = None,
    *,
    initial_episode_id: Array | None = None,
    n_action_samples: int = 1,
    seed: int | None = None,
    n_bootstrap: int = 200,
) -> tuple[float, float]:
    """Estimate initial-state policy value and trajectory-bootstrap SE."""

    values = expected_q_under_policy(q_model, initial_states, target_policy, action_space, n_action_samples, seed)
    mean = float(np.mean(values)) if values.size else float("nan")
    if initial_episode_id is None:
        se = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
    else:
        se = float(_bootstrap_mean_se(values[:, None], np.asarray(initial_episode_id).reshape(-1), n_bootstrap, seed or 0)[0])
    return mean, se


def compute_candidate_logged_q_matrix(
    candidates: Sequence[FQECandidate | Any],
    dataset: TransitionDataset,
    *,
    row_batch_size: int = 4096,
    candidate_batch_size: int | None = None,
    dtype: Any = np.float32,
    memmap_dir: str | Path | None = None,
    max_matrix_bytes: int | None = None,
) -> Array:
    """Compute ``Q_m(obs_i, action_i)`` in chunks without autograd."""

    return _candidate_matrix(
        candidates=candidates,
        n_rows=dataset.n,
        row_batch_size=row_batch_size,
        dtype=dtype,
        memmap_dir=memmap_dir,
        max_matrix_bytes=max_matrix_bytes,
        fn=lambda model, slc, cand_idx: _predict_q(model, dataset.obs[slc], dataset.actions[slc]),
    )


def compute_candidate_next_value_matrix(
    candidates: Sequence[FQECandidate | Any],
    dataset: TransitionDataset,
    target_policy: Any,
    action_space: Any,
    *,
    n_action_samples: int = 1,
    seed: int | None = None,
    apply_terminal: bool = True,
    row_batch_size: int = 4096,
    candidate_batch_size: int | None = None,
    dtype: Any = np.float32,
    memmap_dir: str | Path | None = None,
    max_matrix_bytes: int | None = None,
) -> Array:
    """Compute ``(1-done_i) E_pi Q_m(next_obs_i, .)`` in chunks."""

    del candidate_batch_size  # candidates are adapted one at a time for broad compatibility.
    base_seed = 0 if seed is None else int(seed)

    def predict(model: Any, slc: slice, cand_idx: int) -> Array:
        values = expected_q_under_policy(
            model,
            dataset.next_obs[slc],
            target_policy,
            action_space,
            n_action_samples=n_action_samples,
            seed=base_seed + int(slc.start or 0) + 17,
        )
        if apply_terminal:
            values = (1.0 - dataset.done[slc]) * values
        return values

    return _candidate_matrix(
        candidates=candidates,
        n_rows=dataset.n,
        row_batch_size=row_batch_size,
        dtype=dtype,
        memmap_dir=memmap_dir,
        max_matrix_bytes=max_matrix_bytes,
        fn=predict,
    )


class LowRankOperatorSBVValidator:
    """Low-rank amortized Supervised Bellman Validation for FQE candidates."""

    def __init__(
        self,
        gamma: float,
        ranks: Sequence[int] | int | None = (4, 8, 16, 32),
        *,
        explained_variance: float | None = None,
        max_rank: int | None = None,
        hidden_sizes: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        weight_decay: float = 1e-4,
        n_action_samples: int = 1,
        svd_backend: str = "numpy",
        center_targets: bool = True,
        standardize_coefficients: bool = True,
        device: str = "cpu",
        seed: int = 123,
        row_batch_size: int = 4096,
        candidate_batch_size: int | None = None,
        max_matrix_bytes: int | None = None,
        memmap_dir: str | Path | None = None,
        min_improvement: float = 1e-7,
        n_bootstrap: int = 200,
    ) -> None:
        self.gamma = _validate_gamma(gamma)
        self.ranks = ranks
        self.explained_variance = explained_variance
        self.max_rank = max_rank
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.n_action_samples = int(n_action_samples)
        self.svd_backend = str(svd_backend)
        self.center_targets = bool(center_targets)
        self.standardize_coefficients = bool(standardize_coefficients)
        self.device = str(device)
        self.seed = int(seed)
        self.row_batch_size = int(row_batch_size)
        self.candidate_batch_size = candidate_batch_size
        self.max_matrix_bytes = max_matrix_bytes
        self.memmap_dir = memmap_dir
        self.min_improvement = float(min_improvement)
        self.n_bootstrap = int(n_bootstrap)
        self.trained_operator_model_count = 0
        self.fitted = False

    def fit(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_b_train: TransitionDataset,
        d_b_val: TransitionDataset,
        target_policy: Any,
        action_space: Any,
    ) -> "LowRankOperatorSBVValidator":
        _require_torch()
        _seed_everything(self.seed)
        start = time.perf_counter()
        candidate_list = _as_candidates(candidates)
        H_train = compute_candidate_next_value_matrix(
            candidate_list,
            d_b_train,
            target_policy,
            action_space,
            n_action_samples=self.n_action_samples,
            seed=self.seed + 11,
            row_batch_size=self.row_batch_size,
            candidate_batch_size=self.candidate_batch_size,
            memmap_dir=self.memmap_dir,
            max_matrix_bytes=self.max_matrix_bytes,
        )
        H_val = compute_candidate_next_value_matrix(
            candidate_list,
            d_b_val,
            target_policy,
            action_space,
            n_action_samples=self.n_action_samples,
            seed=self.seed + 11,
            row_batch_size=self.row_batch_size,
            candidate_batch_size=self.candidate_batch_size,
            memmap_dir=self.memmap_dir,
            max_matrix_bytes=self.max_matrix_bytes,
        )
        H_mean = np.mean(H_train, axis=0, keepdims=True) if self.center_targets else np.zeros((1, H_train.shape[1]), dtype=np.float32)
        Hc_train = np.asarray(H_train - H_mean, dtype=np.float32)
        Hc_val = np.asarray(H_val - H_mean, dtype=np.float32)
        max_rank = self._resolve_max_rank(Hc_train)
        Vt_full, singular_values = _truncated_vt(Hc_train, max_rank=max_rank, backend=self.svd_backend, seed=self.seed)
        ranks = self._candidate_ranks(singular_values, max_rank=max_rank)

        x_train, mean, scale = _fit_transform_features(d_b_train.obs, d_b_train.actions, action_space)
        x_val = _transform_features(d_b_val.obs, d_b_val.actions, action_space, mean, scale)
        reward_train = d_b_train.rewards.astype(np.float32)
        reward_val = d_b_val.rewards.astype(np.float32)
        best: dict[str, Any] | None = None
        rank_rows: list[dict[str, Any]] = []
        for rank in ranks:
            Vt = np.asarray(Vt_full[:rank], dtype=np.float32)
            z_train = Hc_train @ Vt.T
            z_val = Hc_val @ Vt.T
            z_mean = np.mean(z_train, axis=0, keepdims=True) if self.standardize_coefficients else np.zeros((1, rank), dtype=np.float32)
            z_scale = np.std(z_train, axis=0, keepdims=True) if self.standardize_coefficients else np.ones((1, rank), dtype=np.float32)
            z_scale = np.where(z_scale > 1e-6, z_scale, 1.0).astype(np.float32)
            train_out = self._train_operator_net(
                x_train=x_train,
                rewards_train=reward_train,
                z_train=((z_train - z_mean) / z_scale).astype(np.float32),
                x_val=x_val,
                rewards_val=reward_val,
                z_val=z_val.astype(np.float32),
                z_mean=z_mean.astype(np.float32),
                z_scale=z_scale.astype(np.float32),
                H_val=np.asarray(H_val, dtype=np.float32),
                H_mean=np.asarray(H_mean, dtype=np.float32),
                Vt=Vt,
                rank=rank,
            )
            self.trained_operator_model_count += 1
            row = {
                "rank": int(rank),
                "operator_val_mse": float(train_out["operator_val_mse"]),
                "reconstruction_mse": float(train_out["reconstruction_mse"]),
                "coefficient_mse": float(train_out["coefficient_mse"]),
                "epochs": int(train_out["epochs"]),
            }
            rank_rows.append(row)
            if best is None or float(row["operator_val_mse"]) < float(best["row"]["operator_val_mse"]):
                best = {
                    "row": row,
                    "network": train_out["network"],
                    "Vt": Vt,
                    "z_mean": z_mean.astype(np.float32),
                    "z_scale": z_scale.astype(np.float32),
                    "rank": int(rank),
                }
        if best is None:
            raise RuntimeError("LowRankOperatorSBVValidator did not train any rank.")
        self.candidates_ = candidate_list
        self.H_mean_ = np.asarray(H_mean, dtype=np.float32)
        self.Vt_ = best["Vt"]
        self.z_mean_ = best["z_mean"]
        self.z_scale_ = best["z_scale"]
        self.rank_ = int(best["rank"])
        self.network_ = best["network"]
        self.input_mean_ = mean
        self.input_scale_ = scale
        self.action_space_ = action_space
        self.diagnostics_ = {
            "rank": int(self.rank_),
            "rank_rows": rank_rows,
            "fit_runtime_sec": float(time.perf_counter() - start),
            "operator_model_count": int(self.trained_operator_model_count),
            "n_candidates": int(len(candidate_list)),
            "n_b_train": int(d_b_train.n),
            "n_b_val": int(d_b_val.n),
            "svd_backend": self.svd_backend,
        }
        self.fitted = True
        return self

    def predict_backup_matrix(self, dataset: TransitionDataset) -> Array:
        if not self.fitted:
            raise RuntimeError("LowRankOperatorSBVValidator must be fit before prediction.")
        x = _transform_features(dataset.obs, dataset.actions, self.action_space_, self.input_mean_, self.input_scale_)
        r_hat, z_hat = _predict_operator(self.network_, x, self.device, self.z_mean_, self.z_scale_)
        H_hat = self.H_mean_ + z_hat @ self.Vt_
        return (r_hat[:, None] + float(self.gamma) * H_hat).astype(np.float64)

    def score(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_score: TransitionDataset,
        target_policy: Any,
        action_space: Any | None = None,
        *,
        initial_states: Array | None = None,
        initial_episode_id: Array | None = None,
    ) -> SBVSelectionResult:
        if not self.fitted:
            raise RuntimeError("LowRankOperatorSBVValidator must be fit before scoring.")
        candidate_list = _as_candidates(candidates)
        resolved_action_space = self.action_space_ if action_space is None else action_space
        _warn_candidate_leakage(candidate_list, d_score.episode_id)
        q_score = compute_candidate_logged_q_matrix(
            candidate_list,
            d_score,
            row_batch_size=self.row_batch_size,
            candidate_batch_size=self.candidate_batch_size,
            memmap_dir=self.memmap_dir,
            max_matrix_bytes=self.max_matrix_bytes,
        ).astype(np.float64)
        backup = self.predict_backup_matrix(d_score)
        residual_sq = (q_score - backup) ** 2
        scores = np.mean(residual_sq, axis=0)
        score_se = _bootstrap_mean_se(residual_sq, d_score.episode_id, self.n_bootstrap, self.seed + 99)
        naive_scores, naive_se = self._naive_td_scores(candidate_list, d_score, target_policy, resolved_action_space)
        rows = _selection_rows(
            method="low_rank_sbv",
            candidates=candidate_list,
            scores=scores,
            score_se=score_se,
            score_key="sbv_score",
            score_se_key="sbv_score_se",
            extra_columns={
                "naive_td_score": naive_scores,
                "naive_td_score_se": naive_se,
            },
            initial_states=initial_states,
            initial_episode_id=initial_episode_id,
            target_policy=target_policy,
            action_space=resolved_action_space,
            n_action_samples=self.n_action_samples,
            seed=self.seed + 1234,
        )
        selected, selected_one_se = _mark_selection(rows, candidate_list, "sbv_score")
        return SBVSelectionResult(
            method="low_rank_sbv",
            rows=rows,
            selected_candidate_id=candidate_list[selected].candidate_id,
            selected_one_se_candidate_id=candidate_list[selected_one_se].candidate_id,
            selected_index=int(selected),
            selected_one_se_index=int(selected_one_se),
            diagnostics=dict(self.diagnostics_),
        )

    def fit_score(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_b_train: TransitionDataset,
        d_b_val: TransitionDataset,
        d_score: TransitionDataset,
        target_policy: Any,
        action_space: Any,
        *,
        initial_states: Array | None = None,
        initial_episode_id: Array | None = None,
    ) -> SBVSelectionResult:
        return self.fit(candidates, d_b_train, d_b_val, target_policy, action_space).score(
            candidates,
            d_score,
            target_policy,
            action_space,
            initial_states=initial_states,
            initial_episode_id=initial_episode_id,
        )

    def _resolve_max_rank(self, Hc_train: Array) -> int:
        hard_max = min(Hc_train.shape)
        if self.max_rank is not None:
            hard_max = min(hard_max, int(self.max_rank))
        if self.ranks is None and self.explained_variance is None:
            return min(hard_max, 8)
        if self.ranks is None:
            return hard_max
        ranks = [int(self.ranks)] if isinstance(self.ranks, int) else [int(rank) for rank in self.ranks]
        if not ranks or any(rank <= 0 for rank in ranks):
            raise ValueError("ranks must contain positive integers.")
        return min(hard_max, max(ranks))

    def _candidate_ranks(self, singular_values: Array, *, max_rank: int) -> list[int]:
        if self.explained_variance is not None:
            threshold = float(self.explained_variance)
            if not (0.0 < threshold <= 1.0):
                raise ValueError("explained_variance must be in (0, 1].")
            energy = np.asarray(singular_values, dtype=np.float64) ** 2
            total = float(np.sum(energy))
            rank = 1 if total <= 0.0 else int(np.searchsorted(np.cumsum(energy) / total, threshold) + 1)
            return [max(1, min(int(rank), int(max_rank)))]
        ranks = [int(self.ranks)] if isinstance(self.ranks, int) else [int(rank) for rank in (self.ranks or (8,))]
        return sorted({max(1, min(int(rank), int(max_rank))) for rank in ranks})

    def _train_operator_net(
        self,
        *,
        x_train: Array,
        rewards_train: Array,
        z_train: Array,
        x_val: Array,
        rewards_val: Array,
        z_val: Array,
        z_mean: Array,
        z_scale: Array,
        H_val: Array,
        H_mean: Array,
        Vt: Array,
        rank: int,
    ) -> dict[str, Any]:
        _require_torch()
        network = _OperatorNet(x_train.shape[1], self.hidden_sizes, int(rank)).to(self.device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_t = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        r_t = torch.as_tensor(rewards_train, dtype=torch.float32, device=self.device).reshape(-1, 1)
        z_t = torch.as_tensor(z_train, dtype=torch.float32, device=self.device)
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        n = x_train.shape[0]
        rng = np.random.default_rng(self.seed + int(rank) * 7919)
        best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        best_metric = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(max(int(self.max_epochs), 1)):
            network.train()
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = torch.as_tensor(order[start : start + self.batch_size], dtype=torch.long, device=self.device)
                pred_r, pred_z = network(x_t[idx])
                loss_r = torch.mean((pred_r.reshape(-1, 1) - r_t[idx]) ** 2)
                loss_z = torch.mean((pred_z - z_t[idx]) ** 2)
                loss = loss_r + loss_z
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            metric_row = _operator_validation_metrics(
                network,
                x_val_t,
                rewards_val,
                z_val,
                z_mean,
                z_scale,
                H_val,
                H_mean,
                Vt,
                self.gamma,
                self.device,
            )
            metric = float(metric_row["operator_val_mse"])
            if metric + self.min_improvement < best_metric:
                best_metric = metric
                best_epoch = int(epoch + 1)
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            else:
                stale += 1
                if self.patience > 0 and stale >= self.patience:
                    break
        network.load_state_dict(best_state)
        final_row = _operator_validation_metrics(
            network,
            x_val_t,
            rewards_val,
            z_val,
            z_mean,
            z_scale,
            H_val,
            H_mean,
            Vt,
            self.gamma,
            self.device,
        )
        final_row.update({"network": network, "epochs": best_epoch})
        return final_row

    def _naive_td_scores(
        self,
        candidates: Sequence[FQECandidate],
        d_score: TransitionDataset,
        target_policy: Any,
        action_space: Any,
    ) -> tuple[Array, Array]:
        q_score = compute_candidate_logged_q_matrix(candidates, d_score, row_batch_size=self.row_batch_size).astype(np.float64)
        H_obs = compute_candidate_next_value_matrix(
            candidates,
            d_score,
            target_policy,
            action_space,
            n_action_samples=self.n_action_samples,
            seed=self.seed + 11,
            row_batch_size=self.row_batch_size,
        ).astype(np.float64)
        target = d_score.rewards[:, None] + float(self.gamma) * H_obs
        residual_sq = (q_score - target) ** 2
        return np.mean(residual_sq, axis=0), _bootstrap_mean_se(residual_sq, d_score.episode_id, self.n_bootstrap, self.seed + 211)


class DirectMultiOutputSBVValidator:
    """Direct multi-output SBV sanity-check baseline for small candidate sets."""

    def __init__(
        self,
        gamma: float,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        weight_decay: float = 1e-4,
        n_action_samples: int = 1,
        direct_threshold: int = 32,
        device: str = "cpu",
        seed: int = 123,
        n_bootstrap: int = 200,
    ) -> None:
        self.gamma = _validate_gamma(gamma)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.n_action_samples = int(n_action_samples)
        self.direct_threshold = int(direct_threshold)
        self.device = str(device)
        self.seed = int(seed)
        self.n_bootstrap = int(n_bootstrap)
        self.trained_model_count = 0
        self.fitted = False

    def fit(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_b_train: TransitionDataset,
        d_b_val: TransitionDataset,
        target_policy: Any,
        action_space: Any,
    ) -> "DirectMultiOutputSBVValidator":
        _require_torch()
        _seed_everything(self.seed)
        candidate_list = _as_candidates(candidates)
        if len(candidate_list) > self.direct_threshold:
            raise ValueError(f"direct multi-output SBV is limited to {self.direct_threshold} candidates.")
        H_train = compute_candidate_next_value_matrix(candidate_list, d_b_train, target_policy, action_space, n_action_samples=self.n_action_samples, seed=self.seed + 5)
        H_val = compute_candidate_next_value_matrix(candidate_list, d_b_val, target_policy, action_space, n_action_samples=self.n_action_samples, seed=self.seed + 5)
        y_train = d_b_train.rewards[:, None] + float(self.gamma) * np.asarray(H_train, dtype=np.float32)
        y_val = d_b_val.rewards[:, None] + float(self.gamma) * np.asarray(H_val, dtype=np.float32)
        x_train, mean, scale = _fit_transform_features(d_b_train.obs, d_b_train.actions, action_space)
        x_val = _transform_features(d_b_val.obs, d_b_val.actions, action_space, mean, scale)
        network = _DirectNet(x_train.shape[1], self.hidden_sizes, len(candidate_list)).to(self.device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x_t = torch.as_tensor(x_train, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        rng = np.random.default_rng(self.seed + 404)
        best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        best_metric = float("inf")
        stale = 0
        for _epoch in range(max(int(self.max_epochs), 1)):
            order = rng.permutation(x_train.shape[0])
            network.train()
            for start in range(0, x_train.shape[0], self.batch_size):
                idx = torch.as_tensor(order[start : start + self.batch_size], dtype=torch.long, device=self.device)
                pred = network(x_t[idx])
                loss = torch.mean((pred - y_t[idx]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                metric = float(torch.mean((network(x_val_t) - y_val_t) ** 2).detach().cpu().item())
            if metric < best_metric - 1e-7:
                best_metric = metric
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            else:
                stale += 1
                if self.patience > 0 and stale >= self.patience:
                    break
        network.load_state_dict(best_state)
        self.network_ = network
        self.input_mean_ = mean
        self.input_scale_ = scale
        self.action_space_ = action_space
        self.candidates_ = candidate_list
        self.diagnostics_ = {"operator_model_count": 1, "direct_val_mse": float(best_metric), "n_candidates": len(candidate_list)}
        self.trained_model_count = 1
        self.fitted = True
        return self

    def predict_backup_matrix(self, dataset: TransitionDataset) -> Array:
        if not self.fitted:
            raise RuntimeError("DirectMultiOutputSBVValidator must be fit before prediction.")
        x = _transform_features(dataset.obs, dataset.actions, self.action_space_, self.input_mean_, self.input_scale_)
        self.network_.eval()
        with torch.no_grad():
            pred = self.network_(torch.as_tensor(x, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
        return pred.astype(np.float64)

    def score(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_score: TransitionDataset,
        target_policy: Any,
        action_space: Any | None = None,
        *,
        initial_states: Array | None = None,
        initial_episode_id: Array | None = None,
    ) -> SBVSelectionResult:
        candidate_list = _as_candidates(candidates)
        resolved_action_space = self.action_space_ if action_space is None else action_space
        q_score = compute_candidate_logged_q_matrix(candidate_list, d_score).astype(np.float64)
        backup = self.predict_backup_matrix(d_score)
        residual_sq = (q_score - backup) ** 2
        scores = np.mean(residual_sq, axis=0)
        se = _bootstrap_mean_se(residual_sq, d_score.episode_id, self.n_bootstrap, self.seed + 909)
        rows = _selection_rows(
            method="direct_sbv",
            candidates=candidate_list,
            scores=scores,
            score_se=se,
            score_key="direct_sbv_score",
            score_se_key="direct_sbv_score_se",
            initial_states=initial_states,
            initial_episode_id=initial_episode_id,
            target_policy=target_policy,
            action_space=resolved_action_space,
            n_action_samples=self.n_action_samples,
            seed=self.seed + 123,
        )
        selected, selected_one_se = _mark_selection(rows, candidate_list, "direct_sbv_score")
        return SBVSelectionResult(
            method="direct_sbv",
            rows=rows,
            selected_candidate_id=candidate_list[selected].candidate_id,
            selected_one_se_candidate_id=candidate_list[selected_one_se].candidate_id,
            selected_index=selected,
            selected_one_se_index=selected_one_se,
            diagnostics=dict(self.diagnostics_),
        )


class GenerativeBellmanValidator:
    """Conditional density-model Bellman validation baseline."""

    def __init__(
        self,
        gamma: float,
        *,
        hidden_sizes: tuple[int, ...] = (128, 128),
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        weight_decay: float = 1e-4,
        n_action_samples: int = 1,
        n_model_samples: int = 16,
        device: str = "cpu",
        seed: int = 123,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        n_bootstrap: int = 200,
    ) -> None:
        self.gamma = _validate_gamma(gamma)
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.n_action_samples = int(n_action_samples)
        self.n_model_samples = int(n_model_samples)
        self.device = str(device)
        self.seed = int(seed)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.n_bootstrap = int(n_bootstrap)
        self.fitted = False

    def fit(
        self,
        d_b_train: TransitionDataset,
        d_b_val: TransitionDataset,
        action_space: Any,
    ) -> "GenerativeBellmanValidator":
        _require_torch()
        _raise_if_nonvector_observation(d_b_train.obs)
        _seed_everything(self.seed)
        x_train, mean, scale = _fit_transform_features(d_b_train.obs, d_b_train.actions, action_space)
        x_val = _transform_features(d_b_val.obs, d_b_val.actions, action_space, mean, scale)
        delta_train = (d_b_train.next_obs - d_b_train.obs).astype(np.float32)
        delta_val = (d_b_val.next_obs - d_b_val.obs).astype(np.float32)
        r_train = d_b_train.rewards.astype(np.float32)
        r_val = d_b_val.rewards.astype(np.float32)
        done_train = d_b_train.done.astype(np.float32)
        done_val = d_b_val.done.astype(np.float32)
        network = _GenerativeNet(x_train.shape[1], self.hidden_sizes, d_b_train.obs.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        train_tensors = (
            torch.as_tensor(x_train, dtype=torch.float32, device=self.device),
            torch.as_tensor(delta_train, dtype=torch.float32, device=self.device),
            torch.as_tensor(r_train, dtype=torch.float32, device=self.device).reshape(-1, 1),
            torch.as_tensor(done_train, dtype=torch.float32, device=self.device).reshape(-1, 1),
        )
        val_tensors = (
            torch.as_tensor(x_val, dtype=torch.float32, device=self.device),
            torch.as_tensor(delta_val, dtype=torch.float32, device=self.device),
            torch.as_tensor(r_val, dtype=torch.float32, device=self.device).reshape(-1, 1),
            torch.as_tensor(done_val, dtype=torch.float32, device=self.device).reshape(-1, 1),
        )
        rng = np.random.default_rng(self.seed + 717)
        best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        best_nll = float("inf")
        initial_nll = _generative_nll(network, val_tensors, self.log_std_min, self.log_std_max, self.device)
        stale = 0
        for _epoch in range(max(int(self.max_epochs), 1)):
            order = rng.permutation(x_train.shape[0])
            network.train()
            x_t, delta_t, r_t, done_t = train_tensors
            for start in range(0, x_train.shape[0], self.batch_size):
                idx = torch.as_tensor(order[start : start + self.batch_size], dtype=torch.long, device=self.device)
                batch = (x_t[idx], delta_t[idx], r_t[idx], done_t[idx])
                loss = _generative_nll_tensor(network, batch, self.log_std_min, self.log_std_max)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            val_nll = _generative_nll(network, val_tensors, self.log_std_min, self.log_std_max, self.device)
            if val_nll < best_nll - 1e-7:
                best_nll = val_nll
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            else:
                stale += 1
                if self.patience > 0 and stale >= self.patience:
                    break
        network.load_state_dict(best_state)
        self.network_ = network
        self.input_mean_ = mean
        self.input_scale_ = scale
        self.action_space_ = action_space
        self.obs_dim_ = d_b_train.obs.shape[1]
        self.diagnostics_ = {
            "initial_validation_nll": float(initial_nll),
            "validation_nll": float(best_nll),
            "operator_model_count": 1,
        }
        self.fitted = True
        return self

    def score(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_score: TransitionDataset,
        target_policy: Any,
        action_space: Any | None = None,
        *,
        initial_states: Array | None = None,
        initial_episode_id: Array | None = None,
    ) -> SBVSelectionResult:
        if not self.fitted:
            raise RuntimeError("GenerativeBellmanValidator must be fit before scoring.")
        candidate_list = _as_candidates(candidates)
        resolved_action_space = self.action_space_ if action_space is None else action_space
        _warn_candidate_leakage(candidate_list, d_score.episode_id)
        q_score = compute_candidate_logged_q_matrix(candidate_list, d_score).astype(np.float64)
        mean_backup, mc_backup = self._backup_matrices(candidate_list, d_score, target_policy, resolved_action_space)
        residual_mc = (q_score - mc_backup) ** 2
        residual_mean = (q_score - mean_backup) ** 2
        scores_mc = np.mean(residual_mc, axis=0)
        se_mc = _bootstrap_mean_se(residual_mc, d_score.episode_id, self.n_bootstrap, self.seed + 313)
        scores_mean = np.mean(residual_mean, axis=0)
        se_mean = _bootstrap_mean_se(residual_mean, d_score.episode_id, self.n_bootstrap, self.seed + 314)
        rows = _selection_rows(
            method="generative",
            candidates=candidate_list,
            scores=scores_mc,
            score_se=se_mc,
            score_key="generative_score_mc",
            score_se_key="generative_score_mc_se",
            extra_columns={"generative_score_mean": scores_mean, "generative_score_mean_se": se_mean},
            initial_states=initial_states,
            initial_episode_id=initial_episode_id,
            target_policy=target_policy,
            action_space=resolved_action_space,
            n_action_samples=self.n_action_samples,
            seed=self.seed + 515,
        )
        selected, selected_one_se = _mark_selection(rows, candidate_list, "generative_score_mc")
        return SBVSelectionResult(
            method="generative",
            rows=rows,
            selected_candidate_id=candidate_list[selected].candidate_id,
            selected_one_se_candidate_id=candidate_list[selected_one_se].candidate_id,
            selected_index=selected,
            selected_one_se_index=selected_one_se,
            diagnostics=dict(self.diagnostics_),
        )

    def fit_score(
        self,
        candidates: Sequence[FQECandidate | Any],
        d_b_train: TransitionDataset,
        d_b_val: TransitionDataset,
        d_score: TransitionDataset,
        target_policy: Any,
        action_space: Any,
        *,
        initial_states: Array | None = None,
        initial_episode_id: Array | None = None,
    ) -> SBVSelectionResult:
        return self.fit(d_b_train, d_b_val, action_space).score(
            candidates,
            d_score,
            target_policy,
            action_space,
            initial_states=initial_states,
            initial_episode_id=initial_episode_id,
        )

    def rollout_value(
        self,
        initial_states: Array,
        target_policy: Any,
        action_space: Any | None = None,
        *,
        horizon: int = 100,
        n_rollouts: int = 1,
        seed: int | None = None,
    ) -> tuple[float, float]:
        if not self.fitted:
            raise RuntimeError("GenerativeBellmanValidator must be fit before rollout.")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        starts = _as_2d_float(initial_states, "initial_states")
        resolved_action_space = self.action_space_ if action_space is None else action_space
        returns = []
        for start_state in starts:
            for _ in range(max(1, int(n_rollouts))):
                obs = start_state.reshape(1, -1).astype(np.float64)
                total = 0.0
                discount = 1.0
                for _step in range(int(horizon)):
                    action = _sample_or_mean_action(target_policy, obs, rng, resolved_action_space)
                    params = self._predict_params(obs, action)
                    delta, reward, done = _sample_generative_params(params, rng, self.log_std_min, self.log_std_max)
                    total += discount * float(reward[0])
                    obs = obs + delta
                    if bool(done[0]):
                        break
                    discount *= float(self.gamma)
                returns.append(total)
        arr = np.asarray(returns, dtype=np.float64)
        se = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        return float(np.mean(arr)), se

    def _backup_matrices(
        self,
        candidates: Sequence[FQECandidate],
        dataset: TransitionDataset,
        target_policy: Any,
        action_space: Any,
    ) -> tuple[Array, Array]:
        params = self._predict_params(dataset.obs, dataset.actions)
        delta_mu, delta_log_std, reward_mu, reward_log_std, done_logit = params
        done_prob = _sigmoid(done_logit.reshape(-1))
        next_obs_mean = dataset.obs + delta_mu
        g_mean_cols = [
            expected_q_under_policy(candidate.model, next_obs_mean, target_policy, action_space, self.n_action_samples, self.seed + 611)
            for candidate in candidates
        ]
        g_mean = np.column_stack(g_mean_cols)
        mean_backup = reward_mu.reshape(-1, 1) + float(self.gamma) * (1.0 - done_prob.reshape(-1, 1)) * g_mean
        rng = np.random.default_rng(self.seed + 612)
        accum = np.zeros_like(mean_backup, dtype=np.float64)
        for _ in range(max(1, int(self.n_model_samples))):
            delta, reward, done = _sample_generative_params(params, rng, self.log_std_min, self.log_std_max)
            next_obs = dataset.obs + delta
            g_cols = [
                expected_q_under_policy(candidate.model, next_obs, target_policy, action_space, self.n_action_samples, self.seed + 613)
                for candidate in candidates
            ]
            g = np.column_stack(g_cols)
            accum += reward.reshape(-1, 1) + float(self.gamma) * (1.0 - done.reshape(-1, 1)) * g
        mc_backup = accum / float(max(1, int(self.n_model_samples)))
        return mean_backup.astype(np.float64), mc_backup.astype(np.float64)

    def _predict_params(self, obs: Array, actions: Array) -> tuple[Array, Array, Array, Array, Array]:
        x = _transform_features(obs, actions, self.action_space_, self.input_mean_, self.input_scale_)
        self.network_.eval()
        with torch.no_grad():
            out = self.network_(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        return tuple(np.asarray(item.detach().cpu().numpy(), dtype=np.float64) for item in out)


def _as_2d_action(actions: Array, name: str, n_rows: int) -> Array:
    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if arr.shape[0] != int(n_rows):
        raise ValueError(f"{name} must have {n_rows} rows.")
    if not np.all(np.isfinite(arr.astype(np.float64))):
        raise ValueError(f"{name} must contain only finite values.")
    return arr.astype(np.float64)


def _as_candidates(candidates: Sequence[FQECandidate | Any]) -> list[FQECandidate]:
    out = []
    for idx, candidate in enumerate(candidates):
        if isinstance(candidate, FQECandidate):
            out.append(candidate)
        else:
            out.append(FQECandidate(candidate_id=f"candidate_{idx:03d}", model=candidate, complexity_order_key=idx))
    if not out:
        raise ValueError("candidates must be nonempty.")
    return out


def _candidate_model(candidate_or_model: FQECandidate | Any) -> Any:
    return candidate_or_model.model if isinstance(candidate_or_model, FQECandidate) else candidate_or_model


def _candidate_matrix(
    *,
    candidates: Sequence[FQECandidate | Any],
    n_rows: int,
    row_batch_size: int,
    dtype: Any,
    memmap_dir: str | Path | None,
    max_matrix_bytes: int | None,
    fn: Callable[[Any, slice, int], Array],
) -> Array:
    candidate_list = _as_candidates(candidates)
    matrix = _allocate_matrix((int(n_rows), len(candidate_list)), dtype=dtype, memmap_dir=memmap_dir, max_matrix_bytes=max_matrix_bytes)
    for cand_idx, candidate in enumerate(candidate_list):
        model = _candidate_model(candidate)
        for start in range(0, int(n_rows), int(row_batch_size)):
            stop = min(start + int(row_batch_size), int(n_rows))
            slc = slice(start, stop)
            matrix[slc, cand_idx] = np.asarray(fn(model, slc, cand_idx), dtype=dtype).reshape(-1)
    return matrix


def _allocate_matrix(shape: tuple[int, int], *, dtype: Any, memmap_dir: str | Path | None, max_matrix_bytes: int | None) -> Array:
    bytes_needed = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if memmap_dir is not None and max_matrix_bytes is not None and bytes_needed > int(max_matrix_bytes):
        root = Path(memmap_dir)
        root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix="fqe_sbv_", suffix=".dat", dir=root, delete=False)
        handle.close()
        return np.memmap(handle.name, dtype=dtype, mode="w+", shape=shape)
    return np.empty(shape, dtype=dtype)


def _discrete_action_values(action_space: Any) -> Array | None:
    if action_space is None:
        return None
    if isinstance(action_space, (int, np.integer)):
        return np.arange(int(action_space), dtype=np.float64).reshape(-1, 1)
    if hasattr(action_space, "n"):
        return np.arange(int(action_space.n), dtype=np.float64).reshape(-1, 1)
    if isinstance(action_space, Mapping) and "actions" in action_space:
        arr = np.asarray(action_space["actions"], dtype=np.float64)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr
    if isinstance(action_space, (list, tuple, np.ndarray)):
        arr = np.asarray(action_space, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
    return None


def _policy_action_probabilities(policy: Any, obs: Array, n_actions: int) -> Array:
    if isinstance(policy, np.ndarray):
        arr = np.asarray(policy, dtype=np.float64)
        if arr.ndim == 1:
            probs = np.repeat(arr.reshape(1, -1), obs.shape[0], axis=0)
        elif arr.ndim == 2:
            state_idx = _state_indices(obs, arr.shape[0])
            probs = arr[state_idx]
        else:
            raise ValueError("discrete policy array must be 1D or 2D.")
    else:
        probs = None
        for name in ("action_probabilities", "predict_proba", "probabilities"):
            if hasattr(policy, name):
                probs = getattr(policy, name)(obs)
                break
        if probs is None and callable(policy):
            probs = policy(obs)
        if probs is None:
            raise ValueError("discrete target_policy must expose action probabilities.")
        probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape != (obs.shape[0], int(n_actions)):
        raise ValueError(f"policy probabilities must have shape ({obs.shape[0]}, {n_actions}).")
    probs = np.maximum(probs, 0.0)
    denom = np.sum(probs, axis=1, keepdims=True)
    if np.any(denom <= 0.0) or not np.all(np.isfinite(probs)):
        raise ValueError("policy probabilities must be finite with positive row sums.")
    return probs / denom


def _state_indices(obs: Array, n_states: int) -> Array:
    if obs.shape[1] == 1:
        idx = obs.reshape(-1).astype(np.int64)
    elif obs.shape[1] == int(n_states):
        idx = np.argmax(obs, axis=1).astype(np.int64)
    else:
        raise ValueError("state-index policy arrays require scalar or one-hot observations.")
    if np.any(idx < 0) or np.any(idx >= int(n_states)):
        raise ValueError("state indices are out of bounds for the policy table.")
    return idx


def _predict_all_actions_if_available(q_model: Any, obs: Array, n_actions: int) -> Array | None:
    model = _candidate_model(q_model)
    for name in ("predict_all_actions", "predict_q_values", "predict_q_all_actions"):
        if hasattr(model, name):
            out = np.asarray(getattr(model, name)(obs), dtype=np.float64)
            if out.shape == (obs.shape[0], int(n_actions)):
                return out
    try:
        out = _call_model(model, obs, None)
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64)
    if arr.shape == (obs.shape[0], int(n_actions)):
        return arr
    return None


def _predict_q(q_model: Any, obs: Array, actions: Array) -> Array:
    model = _candidate_model(q_model)
    obs_2d = _as_2d_float(obs, "obs")
    actions_2d = _as_2d_action(actions, "actions", obs_2d.shape[0])
    if hasattr(model, "predict_q"):
        pred = model.predict_q(obs_2d, actions_2d)
    elif hasattr(model, "predict"):
        try:
            pred = model.predict(obs_2d, actions_2d)
        except TypeError:
            pred = model.predict(np.concatenate([obs_2d, actions_2d], axis=1))
    else:
        pred = _call_model(model, obs_2d, actions_2d)
    out = np.asarray(pred, dtype=np.float64)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out.reshape(-1)
    if out.ndim == 2 and out.shape[0] == obs_2d.shape[0] and actions_2d.shape[1] == 1:
        action_idx = actions_2d.reshape(-1).astype(np.int64)
        if np.all(action_idx >= 0) and np.all(action_idx < out.shape[1]):
            out = out[np.arange(out.shape[0]), action_idx]
    out = out.reshape(-1)
    if out.shape[0] != obs_2d.shape[0]:
        raise ValueError("Q model predictions must have one value per row.")
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Q model produced non-finite predictions.")
    return out.astype(np.float64)


def _call_model(model: Any, obs: Array, actions: Array | None) -> Any:
    if torch is not None and isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            if actions is None:
                return model(obs_t).detach().cpu().numpy()
            actions_t = torch.as_tensor(actions, dtype=torch.float32)
            try:
                return model(obs_t, actions_t).detach().cpu().numpy()
            except TypeError:
                return model(torch.cat([obs_t, actions_t], dim=1)).detach().cpu().numpy()
    if not callable(model):
        raise TypeError("Q model must expose predict_q, predict, or be callable.")
    if actions is None:
        return model(obs)
    try:
        return model(obs, actions)
    except TypeError:
        return model(np.concatenate([obs, actions], axis=1))


def _deterministic_policy_actions(policy: Any, obs: Array) -> Array | None:
    for name in ("deterministic_action", "deterministic_actions", "mean_action", "mean_actions", "predict", "act"):
        if hasattr(policy, name):
            out = getattr(policy, name)(obs)
            return _as_2d_action(out, "policy actions", obs.shape[0])
    if callable(policy):
        out = np.asarray(policy(obs))
        if out.ndim <= 2 and not (out.ndim == 2 and out.shape[1] > 1 and np.allclose(np.sum(out, axis=1), 1.0)):
            return _as_2d_action(out, "policy actions", obs.shape[0])
    return None


def _sample_policy_actions(policy: Any, obs: Array, rng: np.random.Generator, *, sample_idx: int, n_samples: int) -> Array:
    for name in ("sample_actions", "sample"):
        if hasattr(policy, name):
            fn = getattr(policy, name)
            for kwargs in (
                {"rng": rng, "n_samples": n_samples},
                {"rng": rng},
                {"random_state": rng},
                {},
            ):
                try:
                    out = fn(obs, **kwargs)
                    arr = np.asarray(out, dtype=np.float64)
                    if arr.ndim == 3:
                        arr = arr[:, min(sample_idx, arr.shape[1] - 1), :]
                    return _as_2d_action(arr, "sampled actions", obs.shape[0])
                except TypeError:
                    continue
    raise ValueError("stochastic continuous target_policy must expose sample or sample_actions.")


def _sample_or_mean_action(policy: Any, obs: Array, rng: np.random.Generator, action_space: Any) -> Array:
    discrete = _discrete_action_values(action_space)
    if discrete is not None:
        probs = _policy_action_probabilities(policy, obs, discrete.shape[0])
        idx = np.asarray([rng.choice(discrete.shape[0], p=row) for row in probs], dtype=np.int64)
        return discrete[idx]
    deterministic = _deterministic_policy_actions(policy, obs)
    if deterministic is not None:
        return deterministic
    return _sample_policy_actions(policy, obs, rng, sample_idx=0, n_samples=1)


def _fit_transform_features(obs: Array, actions: Array, action_space: Any) -> tuple[Array, Array, Array]:
    raw = _feature_matrix(obs, actions, action_space)
    mean = np.mean(raw, axis=0, keepdims=True)
    scale = np.std(raw, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return ((raw - mean) / scale).astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def _transform_features(obs: Array, actions: Array, action_space: Any, mean: Array, scale: Array) -> Array:
    raw = _feature_matrix(obs, actions, action_space)
    return ((raw - mean) / scale).astype(np.float32)


def _feature_matrix(obs: Array, actions: Array, action_space: Any) -> Array:
    obs_2d = _as_2d_float(obs, "obs")
    action_features = _encode_action(actions, action_space)
    if action_features.shape[0] != obs_2d.shape[0]:
        raise ValueError("obs and actions must have aligned rows.")
    return np.concatenate([obs_2d, action_features], axis=1).astype(np.float64)


def _encode_action(actions: Array, action_space: Any) -> Array:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    discrete = _discrete_action_values(action_space)
    if discrete is None:
        return _as_2d_action(arr, "actions", arr.shape[0])
    n_actions = discrete.shape[0]
    if arr.ndim == 2 and arr.shape[1] == discrete.shape[1] and discrete.shape[1] > 1:
        return arr.astype(np.float64)
    if arr.ndim == 2 and arr.shape[1] == n_actions and np.all((arr >= 0.0) & (arr <= 1.0)):
        return arr.astype(np.float64)
    idx = arr.reshape(-1).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= n_actions):
        if arr.ndim == 2 and arr.shape[1] == discrete.shape[1]:
            return arr.astype(np.float64)
        raise ValueError("discrete action indices are out of bounds.")
    out = np.zeros((idx.shape[0], n_actions), dtype=np.float64)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def _truncated_vt(Hc: Array, *, max_rank: int, backend: str, seed: int) -> tuple[Array, Array]:
    backend = str(backend)
    max_rank = max(1, min(int(max_rank), min(Hc.shape)))
    if backend == "torch":
        _require_torch()
        with torch.no_grad():
            _, s, vt = torch.linalg.svd(torch.as_tensor(Hc, dtype=torch.float32), full_matrices=False)
        return vt[:max_rank].detach().cpu().numpy().astype(np.float32), s[:max_rank].detach().cpu().numpy().astype(np.float64)
    if backend == "randomized" and max_rank < min(Hc.shape):
        rng = np.random.default_rng(int(seed))
        oversample = min(8, max(0, Hc.shape[1] - max_rank))
        omega = rng.normal(size=(Hc.shape[1], max_rank + oversample)).astype(np.float32)
        q, _ = np.linalg.qr(np.asarray(Hc, dtype=np.float32) @ omega, mode="reduced")
        small = q.T @ np.asarray(Hc, dtype=np.float32)
        _, s, vt = np.linalg.svd(small, full_matrices=False)
        return vt[:max_rank].astype(np.float32), s[:max_rank].astype(np.float64)
    if backend != "numpy":
        raise ValueError("svd_backend must be 'randomized', 'torch', or 'numpy'.")
    _, s, vt = np.linalg.svd(np.asarray(Hc, dtype=np.float32), full_matrices=False)
    return vt[:max_rank].astype(np.float32), s[:max_rank].astype(np.float64)


def _bootstrap_mean_se(values: Array, episode_id: Array, n_bootstrap: int, seed: int) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    episodes = np.asarray(episode_id).reshape(-1)
    unique = np.unique(episodes)
    if unique.shape[0] <= 1 or int(n_bootstrap) <= 1:
        return np.zeros(arr.shape[1], dtype=np.float64)
    sums = np.zeros((unique.shape[0], arr.shape[1]), dtype=np.float64)
    counts = np.zeros(unique.shape[0], dtype=np.float64)
    for idx, episode in enumerate(unique):
        mask = episodes == episode
        sums[idx] = np.sum(arr[mask], axis=0)
        counts[idx] = float(np.sum(mask))
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(n_bootstrap), arr.shape[1]), dtype=np.float64)
    for b in range(int(n_bootstrap)):
        pick = rng.integers(0, unique.shape[0], size=unique.shape[0])
        boot[b] = np.sum(sums[pick], axis=0) / max(float(np.sum(counts[pick])), 1e-12)
    return np.std(boot, axis=0, ddof=1)


def _selection_rows(
    *,
    method: str,
    candidates: Sequence[FQECandidate],
    scores: Array,
    score_se: Array,
    score_key: str,
    score_se_key: str,
    extra_columns: Mapping[str, Array] | None = None,
    initial_states: Array | None,
    initial_episode_id: Array | None,
    target_policy: Any,
    action_space: Any,
    n_action_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        value = float("nan")
        value_se = float("nan")
        if initial_states is not None and target_policy is not None:
            value, value_se = estimate_policy_value_from_q(
                candidate.model,
                initial_states,
                target_policy,
                action_space,
                initial_episode_id=initial_episode_id,
                n_action_samples=n_action_samples,
                seed=seed,
            )
        row = {
            "method": method,
            "candidate_id": candidate.candidate_id,
            "fqe_iteration": candidate.fqe_iteration,
            "hyperparams_summary": _summarize_hyperparams(candidate.hyperparams),
            score_key: float(scores[idx]),
            score_se_key: float(score_se[idx]),
            "estimated_policy_value_from_Q": value,
            "estimated_policy_value_se": value_se,
            "selected_min_score": False,
            "selected_one_se": False,
        }
        if extra_columns:
            for key, values in extra_columns.items():
                row[key] = float(np.asarray(values, dtype=np.float64).reshape(-1)[idx])
        rows.append(row)
    return rows


def _mark_selection(rows: list[dict[str, Any]], candidates: Sequence[FQECandidate], score_key: str) -> tuple[int, int]:
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    best = int(np.nanargmin(scores))
    se_key = f"{score_key}_se" if f"{score_key}_se" in rows[best] else score_key.replace("score", "score_se")
    best_se = float(rows[best].get(se_key, 0.0))
    threshold = scores[best] + max(best_se, 0.0)
    eligible = [idx for idx, score in enumerate(scores) if np.isfinite(score) and score <= threshold]
    one_se = min(eligible, key=lambda idx: candidates[idx].complexity_key(idx)) if eligible else best
    rows[best]["selected_min_score"] = True
    rows[one_se]["selected_one_se"] = True
    return best, int(one_se)


def _summarize_hyperparams(hyperparams: Mapping[str, Any] | None) -> str:
    if not hyperparams:
        return ""
    parts = []
    for key in sorted(hyperparams):
        value = hyperparams[key]
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _warn_candidate_leakage(candidates: Sequence[FQECandidate], score_episode_ids: Array) -> None:
    score_ids = set(np.asarray(score_episode_ids).reshape(-1).tolist())
    for candidate in candidates:
        if candidate.trained_on_split_ids is None:
            continue
        overlap = score_ids.intersection(set(candidate.trained_on_split_ids))
        if overlap:
            warnings.warn(
                f"Candidate {candidate.candidate_id} was trained on {len(overlap)} D_score episode id(s); "
                "SBV validation score is no longer clean.",
                RuntimeWarning,
                stacklevel=2,
            )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent.
            torch.cuda.manual_seed_all(int(seed))


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("FQE SBV validators require PyTorch. Install with `pip install -e 'packages/fqe[validation]'`.")


class _OperatorNet(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], rank: int) -> None:
        super().__init__()
        layers: list[Any] = []
        prev = int(input_dim)
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.SiLU())
            prev = int(width)
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.reward_head = nn.Linear(prev, 1)
        self.coeff_head = nn.Linear(prev, int(rank))

    def forward(self, x: Any) -> tuple[Any, Any]:
        h = self.trunk(x)
        return self.reward_head(h).squeeze(-1), self.coeff_head(h)


class _DirectNet(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int) -> None:
        super().__init__()
        layers: list[Any] = []
        prev = int(input_dim)
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.SiLU())
            prev = int(width)
        layers.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.net(x)


class _GenerativeNet(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...], obs_dim: int) -> None:
        super().__init__()
        layers: list[Any] = []
        prev = int(input_dim)
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.SiLU())
            prev = int(width)
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.delta_mu = nn.Linear(prev, int(obs_dim))
        self.delta_log_std = nn.Linear(prev, int(obs_dim))
        self.reward_mu = nn.Linear(prev, 1)
        self.reward_log_std = nn.Linear(prev, 1)
        self.done_logit = nn.Linear(prev, 1)

    def forward(self, x: Any) -> tuple[Any, Any, Any, Any, Any]:
        h = self.trunk(x)
        return self.delta_mu(h), self.delta_log_std(h), self.reward_mu(h), self.reward_log_std(h), self.done_logit(h)


def _predict_operator(network: Any, x: Array, device: str, z_mean: Array, z_scale: Array) -> tuple[Array, Array]:
    network.eval()
    with torch.no_grad():
        r, z_std = network(torch.as_tensor(x, dtype=torch.float32, device=device))
    r_np = r.detach().cpu().numpy().astype(np.float32).reshape(-1)
    z_np = z_std.detach().cpu().numpy().astype(np.float32) * z_scale + z_mean
    return r_np, z_np


def _operator_validation_metrics(
    network: Any,
    x_val_t: Any,
    rewards_val: Array,
    z_val: Array,
    z_mean: Array,
    z_scale: Array,
    H_val: Array,
    H_mean: Array,
    Vt: Array,
    gamma: float,
    device: str,
) -> dict[str, float]:
    del device
    network.eval()
    with torch.no_grad():
        r_hat_t, z_std_t = network(x_val_t)
    r_hat = r_hat_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
    z_hat = z_std_t.detach().cpu().numpy().astype(np.float32) * z_scale + z_mean
    H_hat = H_mean + z_hat @ Vt
    backup_target = rewards_val[:, None] + float(gamma) * H_val
    backup_pred = r_hat[:, None] + float(gamma) * H_hat
    return {
        "operator_val_mse": float(np.mean((backup_target - backup_pred) ** 2)),
        "reconstruction_mse": float(np.mean((H_val - H_hat) ** 2)),
        "coefficient_mse": float(np.mean((z_val - z_hat) ** 2)),
    }


def _generative_nll_tensor(network: Any, batch: tuple[Any, Any, Any, Any], log_std_min: float, log_std_max: float) -> Any:
    x, delta, reward, done = batch
    delta_mu, delta_log_std, reward_mu, reward_log_std, done_logit = network(x)
    delta_log_std = torch.clamp(delta_log_std, float(log_std_min), float(log_std_max))
    reward_log_std = torch.clamp(reward_log_std, float(log_std_min), float(log_std_max))
    delta_var = torch.exp(2.0 * delta_log_std)
    reward_var = torch.exp(2.0 * reward_log_std)
    delta_nll = 0.5 * (((delta - delta_mu) ** 2) / delta_var + 2.0 * delta_log_std + math.log(2.0 * math.pi))
    reward_nll = 0.5 * (((reward - reward_mu) ** 2) / reward_var + 2.0 * reward_log_std + math.log(2.0 * math.pi))
    done_bce = torch.clamp(done_logit, min=0.0) - done_logit * done + torch.log1p(torch.exp(-torch.abs(done_logit)))
    return torch.mean(torch.sum(delta_nll, dim=1, keepdim=True) + reward_nll + done_bce)


def _generative_nll(network: Any, tensors: tuple[Any, Any, Any, Any], log_std_min: float, log_std_max: float, device: str) -> float:
    del device
    network.eval()
    with torch.no_grad():
        return float(_generative_nll_tensor(network, tensors, log_std_min, log_std_max).detach().cpu().item())


def _sample_generative_params(
    params: tuple[Array, Array, Array, Array, Array],
    rng: np.random.Generator,
    log_std_min: float,
    log_std_max: float,
) -> tuple[Array, Array, Array]:
    delta_mu, delta_log_std, reward_mu, reward_log_std, done_logit = params
    delta_std = np.exp(np.clip(delta_log_std, log_std_min, log_std_max))
    reward_std = np.exp(np.clip(reward_log_std, log_std_min, log_std_max))
    delta = delta_mu + delta_std * rng.standard_normal(size=delta_mu.shape)
    reward = reward_mu.reshape(-1) + reward_std.reshape(-1) * rng.standard_normal(size=reward_mu.reshape(-1).shape)
    done_prob = _sigmoid(done_logit.reshape(-1))
    done = (rng.random(size=done_prob.shape) < done_prob).astype(np.float64)
    return delta.astype(np.float64), reward.astype(np.float64), done


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _raise_if_nonvector_observation(obs: Array) -> None:
    arr = np.asarray(obs)
    if arr.ndim != 2:
        raise ValueError("unsupported observation type for default generative baseline: expected 2D vector observations.")
