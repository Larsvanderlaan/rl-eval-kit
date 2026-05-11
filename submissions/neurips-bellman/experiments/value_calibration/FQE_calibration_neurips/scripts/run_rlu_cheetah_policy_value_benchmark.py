#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def _fit_predict(method: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if method == "none":
        return test["raw_fqe_l2"].to_numpy(dtype=float)
    if method == "linear":
        model = LinearRegression().fit(train[["raw_fqe_l2"]], train["official_return"])
        return model.predict(test[["raw_fqe_l2"]])
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(train["raw_fqe_l2"], train["official_return"])
        return model.predict(test["raw_fqe_l2"])
    raise ValueError(f"Unknown method '{method}'")


def _split_indices(name: str, n: int) -> tuple[list[int], list[int]]:
    if name == "early_to_late":
        return list(range(0, n // 2)), list(range(n // 2, n))
    if name == "alternating":
        return list(range(0, n, 2)), list(range(1, n, 2))
    if name == "early5_to_late3":
        return list(range(0, min(5, n))), list(range(min(5, n), n))
    raise ValueError(f"Unknown split '{name}'")


def run(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench = Path(args.benchmark_dir)
    policies = _load_pickle(bench / "rlunplugged_policys.pkl")
    gt = _load_pickle(bench / "rlunplugged_gt.pkl")
    fqe = _load_pickle(bench / "rlunplugged_fqel2.pkl")

    all_ids = list(policies[args.task])
    ids = [all_ids[i] for i in args.policy_indices]
    frame = pd.DataFrame(
        {
            "policy_id": ids,
            "policy_index": list(args.policy_indices),
            "raw_fqe_l2": [float(fqe[pid][0]) for pid in ids],
            "official_return": [float(gt[pid][0]) for pid in ids],
        }
    )

    rows: list[dict[str, object]] = []
    for split_name in args.splits:
        train_idx, test_idx = _split_indices(split_name, len(frame))
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        raw_abs = np.abs(test["raw_fqe_l2"] - test["official_return"])
        raw_mae = float(raw_abs.mean())
        for method in ["none", "linear", "isotonic"]:
            pred = _fit_predict(method, train, test)
            abs_err = np.abs(pred - test["official_return"].to_numpy(dtype=float))
            for i, (_, row) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "task": args.task,
                        "split": split_name,
                        "method": method,
                        "policy_id": row["policy_id"],
                        "policy_index": int(row["policy_index"]),
                        "raw_fqe_l2": float(row["raw_fqe_l2"]),
                        "official_return": float(row["official_return"]),
                        "calibrated_return": float(pred[i]),
                        "absolute_ope_error": float(abs_err[i]),
                        "raw_absolute_ope_error": float(raw_abs.iloc[i]),
                        "relative_absolute_ope_error": float(abs_err[i] / raw_abs.iloc[i]) if raw_abs.iloc[i] > 0 else float("nan"),
                        "calibration_policy_indices": ",".join(str(int(train.iloc[j]["policy_index"])) for j in range(len(train))),
                        "evaluation_policy_indices": ",".join(str(int(test.iloc[j]["policy_index"])) for j in range(len(test))),
                        "data_provenance": "Deep OPE official FQE-L2 score; held-out policy-return calibration",
                    }
                )

    raw_path = out_dir / "rlu_cheetah_policy_value_results.csv"
    summary_path = out_dir / "rlu_cheetah_policy_value_summary.csv"
    audit_path = out_dir / "rlu_cheetah_policy_value_audit.csv"
    config_path = out_dir / "rlu_cheetah_policy_value_config.json"
    _write_csv(rows, raw_path)
    summary = _summary(pd.DataFrame(rows), frame)
    _write_csv(summary, summary_path)
    _write_csv(_audit(summary), audit_path)
    config_path.write_text(json.dumps(vars(args), indent=2, default=str))
    return {"raw": raw_path, "summary": summary_path, "audit": audit_path, "config": config_path}


def _summary(df: pd.DataFrame, full: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    raw_rank = _safe_spearman(full["raw_fqe_l2"], full["official_return"])
    for (split, method), group in df.groupby(["split", "method"]):
        raw = group["raw_absolute_ope_error"].astype(float)
        err = group["absolute_ope_error"].astype(float)
        rows.append(
            {
                "split": split,
                "method": method,
                "n_calibration_policies": int(len(str(group["calibration_policy_indices"].iloc[0]).split(","))),
                "n_evaluation_policies": int(group["policy_id"].nunique()),
                "mean_absolute_ope_error": float(err.mean()),
                "relative_absolute_ope_error": float(err.mean() / raw.mean()) if raw.mean() > 0 else float("nan"),
                "ope_win_rate": float((err < raw).mean()) if method != "none" else float("nan"),
                "raw_policy_spearman": raw_rank,
            }
        )
    return rows


def _audit(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in summary_rows:
        reasons = []
        if row["method"] == "none":
            label = "raw_baseline"
        else:
            if row["raw_policy_spearman"] <= 0.3:
                reasons.append("raw_rank_gate_failed")
            if row["relative_absolute_ope_error"] >= 0.90:
                reasons.append("ope_error_gate_failed")
            if row["ope_win_rate"] < 0.60:
                reasons.append("ope_win_rate_gate_failed")
            label = "promote_main" if not reasons else "not_promoted"
        out.append({**row, "audit_label": label, "failure_reasons": ";".join(reasons)})
    return out


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out policy value calibration on Deep OPE RL Unplugged cheetah_run.")
    parser.add_argument("--benchmark_dir", default="hopper_fqe_benchmark/artifacts/benchmark/dope")
    parser.add_argument("--output_dir", default="FQE_calibration_neurips/results/rlu_cheetah_policy_value_main")
    parser.add_argument("--task", default="cheetah_run")
    parser.add_argument("--policy_indices", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--splits", nargs="+", default=["early_to_late", "alternating", "early5_to_late3"])
    return parser.parse_args()


def main() -> None:
    for name, path in run(parse_args()).items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
