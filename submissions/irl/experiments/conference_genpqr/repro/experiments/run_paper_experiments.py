"""Run the main NeurIPS-ready experimental study with summary tables and plots."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "irl_neurips_paper_repro_cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from data_generation import (
    SimulationConfig,
    deterministic_transition_mean,
    generate_deeppqr_style_data,
    make_anchor_mu,
    make_zero_g,
)
from deeppqr_baseline import fit_deeppqr_baseline
from policy_estimation import EstimatedPolicy, fit_airl_policy, fit_behavior_cloning_policy, fit_maxent_irl_policy
from q_evaluation import fit_fqe_boosted, fit_fqe_neural
from reward_recovery import recover_reward_and_continuation
from utils import EPS




@dataclass(frozen=True)
class MatchedSetting:
    """Configuration for one matched GenPQR-vs-DeepPQR setting."""

    label: str
    n_train_trajectories: int
    anchor_logit_shift: float


@dataclass(frozen=True)
class StudyConfig:
    """Global configuration for the paper experiments."""

    seeds: Sequence[int]
    n_test_trajectories: int = 300
    horizon: int = 10
    n_actions: int = 5
    state_dim: int = 5


class FunctionPolicyAdapter:
    """Wrap a bare policy function for FQE."""

    def __init__(self, policy_fn, n_actions: int):
        self.policy_fn = policy_fn
        self.n_actions = n_actions

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(self.policy_fn(states), dtype=float)

    def sample_actions(self, states: np.ndarray, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(0 if seed is None else seed)
        probs = self.predict_proba(states)
        return np.array([rng.choice(self.n_actions, p=row) for row in probs], dtype=int)


def _build_data(seed: int, n_train_trajectories: int, anchor_logit_shift: float, config: StudyConfig):
    mu = make_anchor_mu(0, config.n_actions)
    g = make_zero_g()
    train_config = SimulationConfig(
        seed=seed,
        horizon=config.horizon,
        n_actions=config.n_actions,
        state_dim=config.state_dim,
        anchor_logit_shift=anchor_logit_shift,
    )
    train = generate_deeppqr_style_data(
        n_trajectories=n_train_trajectories,
        config=train_config,
        mu=mu,
        g=g,
    )
    test_config = SimulationConfig(
        seed=seed + 10_000,
        horizon=config.horizon,
        n_actions=config.n_actions,
        state_dim=config.state_dim,
        anchor_logit_shift=anchor_logit_shift,
    )
    test = generate_deeppqr_style_data(
        n_trajectories=config.n_test_trajectories,
        config=test_config,
        mu=mu,
        g=g,
        simulation_parameters=train["simulation_parameters"],
    )
    return train, test, mu, g, train_config


def _fit_airl(train: Dict[str, np.ndarray], config: SimulationConfig) -> EstimatedPolicy:
    return fit_airl_policy(
        train["states"],
        train["actions"],
        n_actions=config.n_actions,
        next_states=train["next_states"],
        dones=train["dones"],
        gamma=config.gamma,
        transition_model=lambda states, actions: deterministic_transition_mean(
            states=states,
            actions=actions,
            params=train["simulation_parameters"],
        ),
        n_iters=80,
    )


def _reward_metrics(
    method: str,
    reward_pred: np.ndarray,
    test: Dict[str, np.ndarray],
    policy: EstimatedPolicy | None,
    elapsed_sec: float,
    setting_label: str,
    seed: int,
) -> Dict[str, float]:
    actions = test["actions"]
    true_reward = test["normalized_reward_matrix"][np.arange(actions.shape[0]), actions]
    reward_err = reward_pred - true_reward
    reward_corr = float(np.corrcoef(reward_pred, true_reward)[0, 1]) if np.std(true_reward) > 1e-8 else math.nan
    if policy is None:
        policy_nll = math.nan
    else:
        probs = np.clip(policy.predict_proba(test["states"]), EPS, 1.0)
        policy_nll = float(-np.mean(np.log(probs[np.arange(actions.shape[0]), actions])))
    train_anchor_count = float(np.sum(test["trajectory_id"] >= -1))  # placeholder overwritten by caller
    return {
        "seed": float(seed),
        "setting": setting_label,
        "method": method,
        "reward_mse": float(np.mean(reward_err**2)),
        "reward_mae": float(np.mean(np.abs(reward_err))),
        "reward_correlation": reward_corr,
        "policy_nll": policy_nll,
        "elapsed_sec": float(elapsed_sec),
        "mean_reward": float(np.mean(reward_pred)),
        "true_reward_mean": float(np.mean(true_reward)),
        "train_anchor_count": train_anchor_count,
    }


def _attach_counts(row: Dict[str, float], train: Dict[str, np.ndarray]) -> Dict[str, float]:
    row["train_transition_count"] = float(train["actions"].shape[0])
    row["train_anchor_count"] = float(np.sum(train["actions"] == 0))
    row["train_anchor_fraction"] = float(row["train_anchor_count"] / max(row["train_transition_count"], 1.0))
    row["effective_anchor_sample_size"] = row["train_anchor_count"]
    return row


def _evaluate_genpqr(
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    mu,
    g,
    config: SimulationConfig,
    policy: EstimatedPolicy,
    q_kind: str,
    method_name: str,
) -> Dict[str, float]:
    from time import time

    start = time()
    train_probs = np.clip(policy.predict_proba(train["states"]), EPS, 1.0)
    pseudo_rewards = np.log(train_probs[np.arange(train["actions"].shape[0]), train["actions"]]) - train["g_values"]
    mu_policy = FunctionPolicyAdapter(mu, config.n_actions)
    if q_kind == "neural_fqe":
        q_est = fit_fqe_neural(
            states=train["states"],
            actions=train["actions"],
            rewards=pseudo_rewards,
            next_states=train["next_states"],
            dones=train["dones"],
            policy=mu_policy,
            n_actions=config.n_actions,
            gamma=config.gamma,
            n_fqe_iters=8,
            epochs_per_iter=4,
        )
    elif q_kind == "boosted_fqe":
        q_est = fit_fqe_boosted(
            states=train["states"],
            actions=train["actions"],
            rewards=pseudo_rewards,
            next_states=train["next_states"],
            dones=train["dones"],
            policy=mu_policy,
            n_actions=config.n_actions,
            gamma=config.gamma,
            n_fqe_iters=4,
            n_estimators=30,
        )
    else:
        raise ValueError(f"Unknown Q kind: {q_kind}")
    recovered = recover_reward_and_continuation(
        policy_estimate=policy,
        q_estimate=q_est,
        normalization_policy=mu,
        normalization_function=g,
        states=test["states"],
        actions=test["actions"],
        gamma=config.gamma,
    )
    row = _reward_metrics(
        method=method_name,
        reward_pred=recovered["reward"],
        test=test,
        policy=policy,
        elapsed_sec=time() - start,
        setting_label="",
        seed=0,
    )
    return row


def _evaluate_deeppqr(
    train: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    config: SimulationConfig,
    shared_policy: EstimatedPolicy,
) -> Dict[str, float]:
    from time import time

    start = time()
    deeppqr = fit_deeppqr_baseline(
        states=train["states"],
        actions=train["actions"],
        rewards=train["rewards"],
        next_states=train["next_states"],
        dones=train["dones"],
        n_actions=config.n_actions,
        gamma=config.gamma,
        g_values=train["g_values"],
        behavior_policy=shared_policy,
    )
    q_test = deeppqr.full_q.predict_q(test["states"], test["actions"])
    h_test = deeppqr.reward_network.predict(test["states"], test["actions"])
    reward_pred = q_test - config.gamma * h_test
    row = _reward_metrics(
        method="DeepPQR",
        reward_pred=reward_pred,
        test=test,
        policy=shared_policy,
        elapsed_sec=time() - start,
        setting_label="",
        seed=0,
    )
    return row


class LinearRewardBaseline:
    """Strictly linear state-action reward baseline."""

    def __init__(self, theta: np.ndarray, n_actions: int):
        self.theta = theta
        self.n_actions = n_actions

    def _features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=int).reshape(-1)
        intercept = np.ones((states.shape[0], 1), dtype=float)
        shared = np.concatenate([intercept, states], axis=1)
        d = shared.shape[1]
        out = np.zeros((states.shape[0], d * self.n_actions), dtype=float)
        for action in range(self.n_actions):
            mask = actions == action
            if np.any(mask):
                out[mask, action * d : (action + 1) * d] = shared[mask]
        return out

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self._features(states, actions) @ self.theta


def _fit_linear_reward_baseline(train: Dict[str, np.ndarray], n_actions: int, ridge: float = 1e-3) -> LinearRewardBaseline:
    states = train["states"]
    actions = train["actions"]
    rewards = train["rewards"]
    template = LinearRewardBaseline(theta=np.zeros((n_actions * (states.shape[1] + 1),), dtype=float), n_actions=n_actions)
    feats = template._features(states, actions)
    gram = feats.T @ feats + ridge * np.eye(feats.shape[1])
    theta = np.linalg.solve(gram, feats.T @ rewards)
    return LinearRewardBaseline(theta=theta, n_actions=n_actions)


def _evaluate_matched_seed(setting: MatchedSetting, seed: int, study: StudyConfig) -> List[Dict[str, float]]:
    from time import time

    train, test, mu, g, config = _build_data(seed, setting.n_train_trajectories, setting.anchor_logit_shift, study)
    airl_start = time()
    shared_airl = _fit_airl(train, config)
    airl_fit_time = time() - airl_start
    gen = _evaluate_genpqr(train, test, mu, g, config, shared_airl, "neural_fqe", "GenPQR (AIRL, neural FQE)")
    deep = _evaluate_deeppqr(train, test, config, shared_airl)
    gen["elapsed_sec"] += airl_fit_time
    deep["elapsed_sec"] += airl_fit_time
    rows = []
    for row in [gen, deep]:
        row["seed"] = float(seed)
        row["setting"] = setting.label
        row["anchor_logit_shift"] = float(setting.anchor_logit_shift)
        row["n_train_trajectories"] = float(setting.n_train_trajectories)
        row = _attach_counts(row, train)
        rows.append(row)
    return rows


def _evaluate_method_comparison_seed(seed: int, n_train_trajectories: int, anchor_logit_shift: float, study: StudyConfig) -> List[Dict[str, float]]:
    from time import time

    train, test, mu, g, config = _build_data(seed, n_train_trajectories, anchor_logit_shift, study)
    rows: List[Dict[str, float]] = []

    airl_start = time()
    airl_policy = _fit_airl(train, config)
    airl_fit_time = time() - airl_start
    bc_start = time()
    bc_policy = fit_behavior_cloning_policy(train["states"], train["actions"], n_actions=config.n_actions, n_epochs=40)
    bc_fit_time = time() - bc_start
    maxent_start = time()
    maxent_policy = fit_maxent_irl_policy(train["states"], train["actions"], n_actions=config.n_actions, n_iters=150)
    maxent_fit_time = time() - maxent_start

    method_rows = [
        _evaluate_genpqr(train, test, mu, g, config, airl_policy, "neural_fqe", "GenPQR (AIRL, neural FQE)"),
        _evaluate_genpqr(train, test, mu, g, config, airl_policy, "boosted_fqe", "GenPQR (AIRL, boosted FQE)"),
        _evaluate_genpqr(train, test, mu, g, config, bc_policy, "neural_fqe", "GenPQR (BC, neural FQE)"),
        _evaluate_genpqr(train, test, mu, g, config, bc_policy, "boosted_fqe", "GenPQR (BC, boosted FQE)"),
        _evaluate_deeppqr(train, test, config, airl_policy),
    ]
    method_rows[0]["elapsed_sec"] += airl_fit_time
    method_rows[1]["elapsed_sec"] += airl_fit_time
    method_rows[2]["elapsed_sec"] += bc_fit_time
    method_rows[3]["elapsed_sec"] += bc_fit_time
    method_rows[4]["elapsed_sec"] += airl_fit_time

    for row in method_rows:
        row["seed"] = float(seed)
        row["setting"] = "high_sample_comparison"
        row["n_train_trajectories"] = float(n_train_trajectories)
        row["anchor_logit_shift"] = float(anchor_logit_shift)
        rows.append(_attach_counts(row, train))

    actions = test["actions"]
    states = test["states"]

    airl_state_start = time()
    airl_state_reward = airl_policy.predict_state_reward(states)
    row = _reward_metrics(
        method="AIRL state reward",
        reward_pred=airl_state_reward,
        test=test,
        policy=airl_policy,
        elapsed_sec=airl_fit_time + (time() - airl_state_start),
        setting_label="high_sample_comparison",
        seed=seed,
    )
    rows.append(_attach_counts(row, train))

    log_policy_start = time()
    probs = np.clip(airl_policy.predict_proba(states), EPS, 1.0)
    log_policy_reward = np.log(probs[np.arange(actions.shape[0]), actions]) - test["g_values"]
    row = _reward_metrics(
        method="Log-policy pseudo reward",
        reward_pred=log_policy_reward,
        test=test,
        policy=airl_policy,
        elapsed_sec=airl_fit_time + (time() - log_policy_start),
        setting_label="high_sample_comparison",
        seed=seed,
    )
    rows.append(_attach_counts(row, train))

    linear_start = time()
    linear_reward = _fit_linear_reward_baseline(train, config.n_actions)
    row = _reward_metrics(
        method="Linear reward baseline",
        reward_pred=linear_reward.predict(states, actions),
        test=test,
        policy=None,
        elapsed_sec=time() - linear_start,
        setting_label="high_sample_comparison",
        seed=seed,
    )
    rows.append(_attach_counts(row, train))

    maxent_start_eval = time()
    q_maxent = maxent_policy.predict_q(states)
    maxent_reward = q_maxent[np.arange(actions.shape[0]), actions]
    row = _reward_metrics(
        method="MaxEnt-IRL (Q as reward)",
        reward_pred=maxent_reward,
        test=test,
        policy=maxent_policy,
        elapsed_sec=maxent_fit_time + (time() - maxent_start_eval),
        setting_label="high_sample_comparison",
        seed=seed,
    )
    rows.append(_attach_counts(row, train))

    return rows


def _parallel_map(func, items: Sequence, n_jobs: int) -> List:
    if n_jobs <= 1:
        return [func(item) for item in items]
    with get_context("spawn").Pool(processes=n_jobs) as pool:
        return pool.map(func, items)


def _exp1_worker(item, study: StudyConfig):
    setting, seed = item
    return _evaluate_matched_seed(setting, seed, study)


def _flatten(rows: Iterable[Iterable[Dict[str, float]]]) -> pd.DataFrame:
    return pd.DataFrame([row for chunk in rows for row in chunk])


def _summarise(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    summary = (
        df.groupby(list(group_cols), as_index=False)
        .agg(
            n=("seed", "count"),
            reward_mse_mean=("reward_mse", "mean"),
            reward_mse_std=("reward_mse", "std"),
            reward_corr_mean=("reward_correlation", "mean"),
            reward_corr_std=("reward_correlation", "std"),
            policy_nll_mean=("policy_nll", "mean"),
            policy_nll_std=("policy_nll", "std"),
            elapsed_mean=("elapsed_sec", "mean"),
            elapsed_std=("elapsed_sec", "std"),
            anchor_fraction_mean=("train_anchor_fraction", "mean"),
            anchor_count_mean=("train_anchor_count", "mean"),
        )
    )
    for metric in ["reward_mse", "reward_corr", "policy_nll", "elapsed"]:
        std_col = f"{metric}_std"
        mean_col = f"{metric}_mean"
        if std_col in summary:
            summary[f"{metric}_se"] = summary[std_col] / np.sqrt(np.clip(summary["n"], 1, None))
            summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_se"]
    return summary


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_matched(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.4))
    for method, subset in summary.groupby("method"):
        x = subset["anchor_count_mean"].to_numpy()
        order = np.argsort(x)
        x = x[order]
        mse = subset["reward_mse_mean"].to_numpy()[order]
        mse_ci = subset["reward_mse_ci95"].to_numpy()[order]
        ax.errorbar(x, mse, yerr=mse_ci, marker="o", capsize=3, label=method)
    ax.set_xlabel("Effective anchor sample size", fontsize=12)
    ax.set_ylabel("Reward MSE", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "exp1_matched_curve.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_method_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [("reward_mse_mean", "Reward MSE"), ("reward_corr_mean", "Reward correlation"), ("elapsed_mean", "Runtime (s)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    labels = summary["method"].tolist()
    x = np.arange(len(labels))
    for ax, (metric, title) in zip(axes, metrics):
        ci = summary[metric.replace("_mean", "_ci95")].to_numpy()
        ax.bar(x, summary[metric].to_numpy(), yerr=ci, capsize=3, color="#33658A")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_method_comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _save_tables_and_plots(exp1_results: pd.DataFrame, exp1_summary: pd.DataFrame, exp2_results: pd.DataFrame, exp2_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exp1_results.to_csv(output_dir / "exp1_results.csv", index=False)
    exp1_summary.to_csv(output_dir / "exp1_summary.csv", index=False)
    exp2_results.to_csv(output_dir / "exp2_results.csv", index=False)
    exp2_summary.to_csv(output_dir / "exp2_summary.csv", index=False)

    exp1_table = exp1_summary[
        [
            "setting",
            "method",
            "anchor_count_mean",
            "anchor_fraction_mean",
            "reward_mse_mean",
            "reward_mse_ci95",
            "reward_corr_mean",
            "reward_corr_ci95",
            "elapsed_mean",
            "elapsed_ci95",
        ]
    ].copy()
    exp2_table = exp2_summary[
        [
            "method",
            "reward_mse_mean",
            "reward_mse_ci95",
            "reward_corr_mean",
            "reward_corr_ci95",
            "policy_nll_mean",
            "policy_nll_ci95",
            "elapsed_mean",
            "elapsed_ci95",
        ]
    ].copy()
    _write_markdown_table(exp1_table, output_dir / "exp1_table.md")
    _write_markdown_table(exp2_table, output_dir / "exp2_table.md")
    exp1_table.to_csv(output_dir / "exp1_table.csv", index=False)
    exp2_table.to_csv(output_dir / "exp2_table.csv", index=False)
    _plot_matched(exp1_summary, output_dir)
    _plot_method_comparison(exp2_summary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeurIPS paper experiments.")
    parser.add_argument("--replicates", type=int, default=100, help="Number of random seeds.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel worker processes.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "paper_run"),
        help="Directory for outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(range(args.replicates))
    study = StudyConfig(seeds=seeds)
    output_dir = Path(args.output_dir)

    exp1_settings = [
        MatchedSetting(label="low_sample_rare_anchor", n_train_trajectories=200, anchor_logit_shift=-1.0),
        MatchedSetting(label="mid_sample_rare_anchor", n_train_trajectories=1000, anchor_logit_shift=-1.0),
        MatchedSetting(label="high_sample_common_anchor", n_train_trajectories=2500, anchor_logit_shift=0.0),
    ]

    exp1_items = [(setting, seed) for setting in exp1_settings for seed in seeds]
    exp1_func = partial(_exp1_worker, study=study)
    exp1_results = _flatten(_parallel_map(exp1_func, exp1_items, n_jobs=args.jobs))
    exp1_summary = _summarise(exp1_results, ["setting", "method"])

    exp2_items = list(seeds)
    exp2_func = partial(_evaluate_method_comparison_seed, n_train_trajectories=1000, anchor_logit_shift=0.0, study=study)
    exp2_results = _flatten(_parallel_map(exp2_func, exp2_items, n_jobs=args.jobs))
    exp2_summary = _summarise(exp2_results, ["method"])

    _save_tables_and_plots(exp1_results, exp1_summary, exp2_results, exp2_summary, output_dir)
    print(f"Paper experiments completed. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
