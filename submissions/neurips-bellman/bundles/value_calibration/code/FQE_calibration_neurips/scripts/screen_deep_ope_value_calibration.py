#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    train = np.zeros(n, dtype=bool)
    train[order[::2]] = True
    test = ~train
    return train, test


def _evaluate_task(
    family: str,
    task: str,
    policy_ids: list[str],
    gt: dict[str, tuple[float, float]],
    fqe: dict[str, tuple[float, float]],
    seeds: list[int],
) -> list[dict[str, object]]:
    pairs = []
    for policy_id in policy_ids:
        if policy_id in gt and policy_id in fqe:
            pairs.append((policy_id, float(fqe[policy_id][0]), float(gt[policy_id][0])))
    pairs = sorted(pairs, key=lambda x: x[0])
    if len(pairs) < 6:
        return []
    policy = np.array([x[0] for x in pairs], dtype=object)
    raw = np.array([x[1] for x in pairs], dtype=float)
    truth = np.array([x[2] for x in pairs], dtype=float)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train, test = _split_indices(len(pairs), seed)
        if np.unique(raw[train]).size < 2:
            continue
        methods = {
            "raw_official_fqe_l2": raw.copy(),
        }
        linear = LinearRegression().fit(raw[train, None], truth[train])
        methods["linear_policy_value_calibration"] = linear.predict(raw[:, None])
        isotonic = IsotonicRegression(out_of_bounds="clip").fit(raw[train], truth[train])
        methods["isotonic_policy_value_calibration"] = isotonic.predict(raw)
        raw_mae = float(np.mean(np.abs(raw[test] - truth[test])))
        for method, pred in methods.items():
            mae = float(np.mean(np.abs(pred[test] - truth[test])))
            rows.append(
                {
                    "family": family,
                    "task": task,
                    "seed": int(seed),
                    "method": method,
                    "n_policies": int(len(pairs)),
                    "n_train_policies": int(train.sum()),
                    "n_test_policies": int(test.sum()),
                    "raw_policy_spearman": _safe_spearman(raw, truth),
                    "test_policy_ids": ";".join(policy[test].astype(str).tolist()),
                    "mean_absolute_error": mae,
                    "relative_absolute_error_vs_raw": mae / raw_mae if raw_mae > 0 else float("nan"),
                }
            )
    return rows


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    out = []
    for (family, method), group in df.groupby(["family", "method"], dropna=False):
        rel = pd.to_numeric(group["relative_absolute_error_vs_raw"], errors="coerce")
        spearman = pd.to_numeric(group["raw_policy_spearman"], errors="coerce")
        out.append(
            {
                "family": family,
                "method": method,
                "n_task_seed_rows": int(len(group)),
                "n_tasks": int(group["task"].nunique()),
                "median_raw_policy_spearman": float(spearman.median()),
                "median_relative_absolute_error_vs_raw": float(rel.median()),
                "mean_relative_absolute_error_vs_raw": float(rel.mean()),
                "win_rate_vs_raw": float((rel < 1.0).mean()),
                "strong_win_rate_vs_raw": float((rel < 0.9).mean()),
            }
        )
    return out


def _audit(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in summary_rows:
        method = str(row["method"])
        if method == "raw_official_fqe_l2":
            label = "raw_baseline"
            reasons = []
        else:
            reasons = []
            if float(row["median_raw_policy_spearman"]) <= 0:
                reasons.append("raw_not_informative")
            if float(row["median_relative_absolute_error_vs_raw"]) >= 0.9:
                reasons.append("median_error_gate_failed")
            if float(row["win_rate_vs_raw"]) < 0.6:
                reasons.append("win_rate_gate_failed")
            label = "policy_value_calibration_benchmark" if not reasons else "not_promoted"
        rows.append({**row, "audit_label": label, "failure_reasons": ";".join(reasons)})
    return rows


def _plot(summary_rows: list[dict[str, object]], output_dir: Path) -> None:
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return
    df = df[df["method"].astype(str).ne("raw_official_fqe_l2")].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.4), constrained_layout=True)
    labels = df["family"].astype(str) + " / " + df["method"].astype(str).str.replace("_policy_value_calibration", "", regex=False)
    ax.barh(labels, df["median_relative_absolute_error_vs_raw"], color=["#4C78A8", "#59A14F"] * 4)
    ax.axvline(1.0, color="0.35", linewidth=0.8)
    ax.set_xlabel("median relative policy-value MAE")
    ax.set_title("Deep OPE official FQE-L2 value calibration screen", loc="left", fontweight="bold")
    fig.savefig(output_dir / "deep_ope_value_calibration_screen.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "deep_ope_value_calibration_screen.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Path]:
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for family in args.families:
        gt = _load_pickle(benchmark_dir / f"{family}_gt.pkl")
        fqe = _load_pickle(benchmark_dir / f"{family}_fqel2.pkl")
        policies = _load_pickle(benchmark_dir / f"{family}_policys.pkl")
        for task, policy_ids in policies.items():
            rows.extend(_evaluate_task(str(family), str(task), list(policy_ids), gt, fqe, list(args.seeds)))
    summary = _summarize(rows)
    audit = _audit(summary)
    raw_path = output_dir / "deep_ope_value_calibration_rows.csv"
    summary_path = output_dir / "deep_ope_value_calibration_summary.csv"
    audit_path = output_dir / "deep_ope_value_calibration_audit.csv"
    config_path = output_dir / "deep_ope_value_calibration_config.json"
    _write_csv(rows, raw_path)
    _write_csv(summary, summary_path)
    _write_csv(audit, audit_path)
    config_path.write_text(json.dumps(vars(args), indent=2))
    _plot(summary, output_dir)
    return {"raw": raw_path, "summary": summary_path, "audit": audit_path, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen Deep OPE official FQE-L2 predictions for post-hoc value calibration.")
    parser.add_argument("--benchmark_dir", default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--output_dir", default="FQE_calibration_neurips/results/deep_ope_value_calibration_screen")
    parser.add_argument("--families", nargs="+", default=["d4rl", "rlunplugged"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    for name, path in run(args).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
