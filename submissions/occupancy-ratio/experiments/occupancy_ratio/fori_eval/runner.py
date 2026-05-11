from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.occupancy_ratio.fori_eval.ablations import run_population_ablations
from experiments.occupancy_ratio.fori_eval.estimators import (
    EstimatorOutput,
    run_boosted_tree,
    run_google_dualdice_sample,
    run_neural_network,
    run_oracle,
    run_population_fori,
)
from experiments.occupancy_ratio.fori_eval.finite_mdp import make_random_finite_mdp, sample_finite_dataset
from experiments.occupancy_ratio.fori_eval.metrics import evaluate_grid_weights, evaluate_sample_weights
from experiments.occupancy_ratio.fori_eval.plots import write_plots


@dataclass(frozen=True)
class RunnerConfig:
    profile: str = "smoke"
    output_root: Path = Path("outputs/fori_eval")
    run_id: str | None = None
    estimators: tuple[str, ...] = ("oracle", "population_fori", "boosted_tree_stable", "neural_network_stable")
    seeds: tuple[int, ...] = (0,)
    sample_sizes: tuple[int, ...] = (300,)
    gammas: tuple[float, ...] = (0.9,)
    n_states: tuple[int, ...] = (20,)
    n_actions: tuple[int, ...] = (2,)
    transition_concentrations: tuple[float, ...] = (1.0,)
    mismatches: tuple[float, ...] = (1.0,)
    overlap_floor: float = 0.02
    reward_sweeps: int = 64
    population_iterations: int = 30
    external_repo_path: str = "/tmp/google-research"
    write_plots: bool = True

    @classmethod
    def for_profile(cls, profile: str, **overrides: Any) -> "RunnerConfig":
        base: dict[str, Any]
        if profile == "smoke":
            base = {}
        elif profile == "medium":
            base = dict(
                seeds=(0, 1, 2),
                sample_sizes=(500, 1500),
                gammas=(0.5, 0.9, 0.95),
                n_states=(20, 50),
                n_actions=(2, 5),
                transition_concentrations=(0.1, 1.0, 10.0),
                mismatches=(0.0, 0.75, 1.5),
                population_iterations=50,
            )
        elif profile == "full":
            base = dict(
                seeds=tuple(range(10)),
                sample_sizes=(500, 1500, 5000),
                gammas=(0.5, 0.8, 0.9, 0.95, 0.99),
                n_states=(20, 50, 100),
                n_actions=(2, 5),
                transition_concentrations=(0.1, 1.0, 10.0),
                mismatches=(0.0, 0.75, 1.5, 2.5),
                reward_sweeps=128,
                population_iterations=80,
            )
        else:
            raise ValueError("profile must be smoke, medium, or full.")
        base.update(overrides)
        return cls(profile=profile, **base)

    def output_dir(self) -> Path:
        run_id = self.run_id or time.strftime("%Y%m%d_%H%M%S")
        return self.output_root / run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-local FORI finite-MDP experiments.")
    parser.add_argument("--profile", choices=("smoke", "medium", "full"), default="smoke")
    parser.add_argument("--output-root", default="outputs/fori_eval")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--n-states", nargs="*", type=int, default=None)
    parser.add_argument("--n-actions", nargs="*", type=int, default=None)
    parser.add_argument("--transition-concentrations", nargs="*", type=float, default=None)
    parser.add_argument("--mismatches", nargs="*", type=float, default=None)
    parser.add_argument("--overlap-floor", type=float, default=None)
    parser.add_argument("--population-iterations", type=int, default=None)
    parser.add_argument("--external-repo-path", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updates: dict[str, Any] = {
        "output_root": Path(args.output_root),
        "write_plots": not args.no_plots,
    }
    for key, value in (
        ("run_id", args.run_id),
        ("estimators", args.estimators),
        ("seeds", args.seeds),
        ("sample_sizes", args.sample_sizes),
        ("gammas", args.gammas),
        ("n_states", args.n_states),
        ("n_actions", args.n_actions),
        ("transition_concentrations", args.transition_concentrations),
        ("mismatches", args.mismatches),
        ("overlap_floor", args.overlap_floor),
        ("population_iterations", args.population_iterations),
        ("external_repo_path", args.external_repo_path),
    ):
        if value is not None:
            updates[key] = tuple(value) if isinstance(value, list) else value
    config = RunnerConfig.for_profile(args.profile, **updates)
    result = run(config)
    print(f"Wrote results: {result['results_path']}")
    print(f"Wrote summary: {result['summary_path']}")
    print(f"Wrote diagnostics: {result['diagnostics_path']}")
    if result["plot_paths"]:
        print("Wrote plots:")
        for path in result["plot_paths"]:
            print(f"  {path}")


def run(config: RunnerConfig) -> dict[str, Any]:
    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    estimators = expand_estimators(config.estimators)

    for seed in config.seeds:
        for n_states in config.n_states:
            for n_actions in config.n_actions:
                for concentration in config.transition_concentrations:
                    for mismatch in config.mismatches:
                        mdp_seed = stable_seed(seed, n_states, n_actions, concentration, mismatch)
                        mdp = make_random_finite_mdp(
                            n_states=n_states,
                            n_actions=n_actions,
                            transition_concentration=concentration,
                            mismatch=mismatch,
                            overlap_floor=config.overlap_floor,
                            seed=mdp_seed,
                        )
                        for sample_size in config.sample_sizes:
                            for gamma in config.gammas:
                                dataset = sample_finite_dataset(
                                    mdp=mdp,
                                    gamma=gamma,
                                    sample_size=sample_size,
                                    seed=stable_seed(seed, sample_size, gamma, 17),
                                    n_reward_sweeps=config.reward_sweeps,
                                )
                                for estimator in estimators:
                                    outputs = run_one_estimator(estimator, dataset, config, seed)
                                    for output in outputs:
                                        row = row_from_output(config, dataset, output)
                                        rows.append(row)
                                        key = row_key(row)
                                        if output.history:
                                            histories[key] = output.history
                                        if output.status == "error":
                                            failures.append({"estimator": output.name, "reason": output.skip_reason})

    results_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.csv"
    diagnostics_path = output_dir / "diagnostics.json"
    manifest_path = output_dir / "manifest.json"
    write_csv(results_path, rows)
    summary = summarize(rows)
    write_csv(summary_path, summary)
    write_json(
        diagnostics_path,
        {
            "failures": failures,
            "histories": histories,
            "n_rows": len(rows),
        },
    )
    write_json(manifest_path, {"config": jsonable(asdict(config)), "estimators": estimators})
    plot_paths = write_plots(output_dir, rows, histories) if config.write_plots else []
    return {
        "output_dir": output_dir,
        "results_path": results_path,
        "summary_path": summary_path,
        "diagnostics_path": diagnostics_path,
        "manifest_path": manifest_path,
        "plot_paths": plot_paths,
    }


def run_one_estimator(estimator: str, dataset, config: RunnerConfig, seed: int) -> list[EstimatorOutput]:
    if estimator == "oracle":
        return [run_oracle(dataset)]
    if estimator in {"population", "population_fori"}:
        return [run_population_fori(dataset, num_iterations=config.population_iterations)]
    if estimator == "ablations":
        return run_population_ablations(dataset, num_iterations=config.population_iterations)
    if estimator.startswith("boosted_tree_"):
        return [run_boosted_tree(dataset, preset=estimator.removeprefix("boosted_tree_"), profile=config.profile, seed=seed)]
    if estimator.startswith("neural_network_"):
        return [run_neural_network(dataset, preset=estimator.removeprefix("neural_network_"), profile=config.profile, seed=seed)]
    if estimator == "google_dualdice_neural":
        return [run_google_dualdice_sample(dataset, external_repo_path=config.external_repo_path, seed=seed)]
    raise ValueError(f"Unknown estimator '{estimator}'.")


def row_from_output(config: RunnerConfig, dataset, output: EstimatorOutput) -> dict[str, Any]:
    row: dict[str, Any] = {
        "profile": config.profile,
        "setting": dataset.setting,
        "estimator": output.name,
        "status": output.status,
        "skip_reason": output.skip_reason,
        "scope": output.scope,
        "runtime_sec": output.runtime_sec,
        "seed": dataset.seed,
        "sample_size": dataset.sample_size,
        "gamma": dataset.truth.gamma,
        **dataset.mdp.metadata,
    }
    if output.status == "ok" and output.weights is not None:
        if output.scope == "grid":
            row.update(evaluate_grid_weights(dataset=dataset, weights=output.weights, raw_weights=output.raw_weights))
        else:
            row.update(evaluate_sample_weights(dataset=dataset, weights=output.weights, raw_weights=output.raw_weights))
    row.update(output.diagnostics)
    return jsonable(row)


def expand_estimators(estimators: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for name in estimators:
        if name == "boosted":
            out.extend(["boosted_tree_squared", "boosted_tree_huber", "boosted_tree_stable"])
        elif name == "neural":
            out.extend(["neural_network_squared", "neural_network_huber", "neural_network_stable"])
        elif name == "boosted_sensitivity":
            out.extend([
                "boosted_tree_squared",
                "boosted_tree_huber",
                "boosted_tree_stable",
                "boosted_tree_transition_norm",
                "boosted_tree_calibrated",
                "boosted_tree_stable_logistic_nuisance",
            ])
        elif name == "neural_sensitivity":
            out.extend([
                "neural_network_squared",
                "neural_network_huber",
                "neural_network_stable",
                "neural_network_transition_norm",
                "neural_network_calibrated",
                "neural_network_stable_logistic_nuisance",
            ])
        elif name == "dice":
            out.append("google_dualdice_neural")
        else:
            out.append(str(name))
    return tuple(dict.fromkeys(out))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("profile"),
            row.get("setting"),
            row.get("estimator"),
            row.get("status"),
            row.get("gamma"),
            row.get("sample_size"),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary_row = {
            "profile": key[0],
            "setting": key[1],
            "estimator": key[2],
            "status": key[3],
            "gamma": key[4],
            "sample_size": key[5],
            "n_runs": len(group),
        }
        numeric = sorted(
            {
                name
                for row in group
                for name, value in row.items()
                if is_finite_number(value)
            }
        )
        for name in numeric:
            vals = np.asarray([float(row[name]) for row in group if is_finite_number(row.get(name))])
            if vals.size:
                summary_row[f"{name}_mean"] = float(np.mean(vals))
                summary_row[f"{name}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append(summary_row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def row_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("estimator"),
        row.get("seed"),
        row.get("sample_size"),
        row.get("gamma"),
        row.get("n_states"),
        row.get("n_actions"),
        row.get("transition_concentration"),
        row.get("mismatch"),
    ]
    return "__".join(str(part) for part in parts)


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    acc = 1729
    for char in text:
        acc = (acc * 131 + ord(char)) % (2**32 - 1)
    return int(acc)


def is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(jsonable(value), sort_keys=True)
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
