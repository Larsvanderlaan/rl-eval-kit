from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from causal_ope_benchmark.api import list_difficulties, list_estimators, list_families
from causal_ope_benchmark.calibration import CalibrationStudyConfig, run_calibration_study
from causal_ope_benchmark.config import CausalOPEBenchmarkConfig
from causal_ope_benchmark.exceptions import ConfigurationError
from causal_ope_benchmark.policies import POLICY_NAMES_BY_FAMILY
from causal_ope_benchmark.runner import run_benchmark
from causal_ope_benchmark.stress import DifficultyStressStudyConfig, run_difficulty_stress_study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realistic causal OPE benchmarks.")
    parser.add_argument("--profile", choices=("smoke", "core", "full", "paper"), default="smoke")
    parser.add_argument("--output-root", default="outputs/causal_ope_benchmark")
    parser.add_argument("--families", nargs="*", choices=("streamlift", "streamretain", "clinic_dtr", "epicare"), default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--observed-horizons", nargs="*", type=int, default=None)
    parser.add_argument("--target-policies", nargs="*", default=None)
    parser.add_argument("--mc-truth-rollouts", type=int, default=None)
    parser.add_argument("--streamlift-include-infinite-horizon", action="store_true", help="Add discounted infinite-horizon StreamLift estimands.")
    parser.add_argument("--streamlift-infinite-horizon-max-steps", type=int, default=None)
    parser.add_argument("--automl-tuning", choices=("off", "fast", "balanced"), default=None)
    parser.add_argument("--epicare-core-pilot", action="store_true", help="Use the EpiCare tree-vs-neural core pilot matrix.")
    parser.add_argument("--list-families", action="store_true", help="List registered families and exit.")
    parser.add_argument("--list-estimators", action="store_true", help="List registered estimators and exit.")
    parser.add_argument("--list-difficulties", action="store_true", help="List registered difficulty profiles and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without running simulations.")
    parser.add_argument("--verbose", action="store_true", help="Print a compact run summary after completion.")
    return parser.parse_args(argv)


def parse_calibration_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural FQE and occupancy-ratio calibration screens.")
    parser.add_argument("--preset", choices=("smoke", "core-lite", "full"), default="core-lite")
    parser.add_argument("--output-root", default="outputs/causal_ope_benchmark")
    parser.add_argument("--families", nargs="*", choices=("streamlift", "streamretain", "clinic_dtr", "epicare"), default=None)
    parser.add_argument("--include-epicare", action="store_true", help="Include the optional external EpiCare family.")
    parser.add_argument("--include-stress", action="store_true", help="Include sensitivity/stress cells in addition to primary calibration cells.")
    parser.add_argument("--estimators", nargs="*", choices=("neural_fqe", "neural_occupancy"), default=None)
    parser.add_argument("--tuning-tracks", nargs="*", choices=("proxy", "oracle"), default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--target-policies", nargs="*", default=None)
    parser.add_argument("--mc-truth-rollouts", type=int, default=None)
    parser.add_argument("--trajectory-horizon", type=int, default=None)
    parser.add_argument("--fqe-budget", choices=("fast", "balanced"), default=None)
    parser.add_argument("--occupancy-budget", choices=("fast", "balanced"), default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved calibration configuration without running.")
    parser.add_argument("--verbose", action="store_true", help="Print a compact calibration summary after completion.")
    return parser.parse_args(argv)


def parse_stress_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run systematic difficulty stress tests.")
    parser.add_argument("--scale", choices=("ci", "audit", "exhaustive"), default="ci")
    parser.add_argument("--output-root", default="outputs/causal_ope_benchmark")
    parser.add_argument("--difficulty", "--difficulties", dest="difficulties", nargs="*", choices=("easy", "medium", "hard", "realistic"), default=None)
    parser.add_argument("--families", nargs="*", choices=("streamlift", "streamretain", "clinic_dtr", "epicare"), default=None)
    parser.add_argument("--include-epicare", action="store_true", help="Include the optional external EpiCare family.")
    parser.add_argument("--include-sensitivity", action="store_true", help="Include assumption-violation sensitivity cells.")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--gammas", nargs="*", type=float, default=None)
    parser.add_argument("--target-policies", nargs="*", default=None)
    parser.add_argument("--mc-truth-rollouts", type=int, default=None)
    parser.add_argument("--google-research-path", default=None)
    parser.add_argument("--dice-rl-repo-path", default=None)
    parser.add_argument("--no-oracle-tracks", action="store_true", help="Disable diagnostic oracle-tuned neural rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved stress-study configuration without running.")
    parser.add_argument("--verbose", action="store_true", help="Print a compact stress-study summary after completion.")
    return parser.parse_args(argv)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        _main_calibrate(parse_calibration_args(sys.argv[2:]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "stress-test":
        _main_stress(parse_stress_args(sys.argv[2:]))
        return
    args = parse_args()
    if args.list_families:
        for family in list_families():
            default = "default" if family.default_profile_member else "opt-in"
            print(f"{family.name}\t{family.display_name}\t{default}\t{family.summary}")
        return
    if args.list_estimators:
        for estimator in list_estimators():
            diagnostic = "diagnostic" if estimator.diagnostic_only else "primary"
            optional = estimator.optional_dependency or "none"
            print(f"{estimator.name}\t{diagnostic}\toptional={optional}\t{estimator.summary}")
        return
    if args.list_difficulties:
        for difficulty in list_difficulties():
            low, high = difficulty.target_policy_distance_range
            print(f"{difficulty.name}\toverlap={difficulty.overlap}\ttarget_tv={low:.2f}-{high:.2f}\t{difficulty.summary}")
        return
    try:
        if args.epicare_core_pilot:
            config = CausalOPEBenchmarkConfig.epicare_core_pilot(output_root=Path(args.output_root))
        else:
            config = CausalOPEBenchmarkConfig.for_profile(args.profile, output_root=Path(args.output_root))
        updates = {}
        for key, value in (
            ("families", args.families),
            ("estimators", args.estimators),
            ("seeds", args.seeds),
            ("sample_sizes", args.sample_sizes),
            ("gammas", args.gammas),
            ("observed_horizons", args.observed_horizons),
            ("target_policies", args.target_policies),
            ("mc_truth_rollouts", args.mc_truth_rollouts),
            ("streamlift_infinite_horizon_max_steps", args.streamlift_infinite_horizon_max_steps),
            ("automl_tuning", args.automl_tuning),
        ):
            if value is not None:
                updates[key] = tuple(value) if isinstance(value, list) else value
        if args.streamlift_include_infinite_horizon:
            updates["streamlift_include_infinite_horizon"] = True
        if updates:
            config = CausalOPEBenchmarkConfig(**{**config.__dict__, **updates})
        _validate_cli_config(config, target_policies_supplied=args.target_policies is not None)
    except (ConfigurationError, ValueError) as exc:
        _exit_with_error(str(exc))
    if args.dry_run:
        print(json.dumps(_config_payload(config), indent=2, sort_keys=True, default=str))
        return
    result = run_benchmark(config)
    print(f"wrote results to {result.results_path}")
    print(f"wrote summary to {result.summary_path}")
    print(f"wrote schema to {result.output_schema_path}")
    print(f"wrote readout to {result.readout_path}")
    if args.verbose:
        ok_rows = sum(1 for row in result.rows if row.get("status") == "ok")
        failed_rows = len(result.rows) - ok_rows
        print(f"rows={len(result.rows)} ok={ok_rows} non_ok={failed_rows} summary_rows={len(result.summary_rows)}")


def _main_calibrate(args: argparse.Namespace) -> None:
    try:
        config = CalibrationStudyConfig.for_preset(args.preset, output_root=Path(args.output_root))
        updates = {}
        for key, value in (
            ("families", args.families),
            ("estimators", args.estimators),
            ("tuning_tracks", args.tuning_tracks),
            ("seeds", args.seeds),
            ("sample_sizes", args.sample_sizes),
            ("gammas", args.gammas),
            ("target_policies", args.target_policies),
            ("mc_truth_rollouts", args.mc_truth_rollouts),
            ("trajectory_horizon", args.trajectory_horizon),
            ("fqe_budget", args.fqe_budget),
            ("occupancy_budget", args.occupancy_budget),
        ):
            if value is not None:
                updates[key] = tuple(value) if isinstance(value, list) else value
        if args.include_epicare:
            updates["include_epicare"] = True
        if args.include_stress:
            updates["include_stress"] = True
        if updates:
            config = CalibrationStudyConfig(**{**config.__dict__, **updates})
        _validate_target_policies(tuple(config.families), tuple(config.target_policies or ()), target_policies_supplied=args.target_policies is not None)
    except (ConfigurationError, ValueError) as exc:
        _exit_with_error(str(exc))
    if args.dry_run:
        print(json.dumps(_calibration_config_payload(config), indent=2, sort_keys=True, default=str))
        return
    result = run_calibration_study(config)
    print(f"wrote calibration results to {result.results_path}")
    print(f"wrote calibration summary to {result.summary_path}")
    print(f"wrote calibration candidates to {result.candidates_path}")
    print(f"wrote calibration readout to {result.readout_path}")
    if args.verbose:
        ok_rows = sum(1 for row in result.rows if row.get("status") == "ok")
        failed_rows = len(result.rows) - ok_rows
        print(f"rows={len(result.rows)} ok={ok_rows} non_ok={failed_rows} candidate_rows={len(result.candidate_rows)}")


def _main_stress(args: argparse.Namespace) -> None:
    try:
        config = DifficultyStressStudyConfig.for_scale(args.scale, output_root=Path(args.output_root))
        updates = {}
        for key, value in (
            ("difficulties", args.difficulties),
            ("families", args.families),
            ("methods", args.methods),
            ("seeds", args.seeds),
            ("sample_sizes", args.sample_sizes),
            ("gammas", args.gammas),
            ("target_policies", args.target_policies),
            ("mc_truth_rollouts", args.mc_truth_rollouts),
            ("google_research_path", args.google_research_path),
            ("dice_rl_repo_path", args.dice_rl_repo_path),
        ):
            if value is not None:
                updates[key] = tuple(value) if isinstance(value, list) else value
        if args.include_epicare:
            updates["include_epicare"] = True
        if args.include_sensitivity:
            updates["include_sensitivity"] = True
        if args.no_oracle_tracks:
            updates["include_oracle_tracks"] = False
        if updates:
            config = DifficultyStressStudyConfig(**{**config.__dict__, **updates})
        _validate_target_policies(tuple(config.families), tuple(config.target_policies or ()), target_policies_supplied=args.target_policies is not None)
    except (ConfigurationError, ValueError) as exc:
        _exit_with_error(str(exc))
    if args.dry_run:
        print(json.dumps(_stress_config_payload(config), indent=2, sort_keys=True, default=str))
        return
    result = run_difficulty_stress_study(config)
    print(f"wrote difficulty results to {result.results_path}")
    print(f"wrote difficulty summary to {result.summary_path}")
    print(f"wrote difficulty candidates to {result.candidates_path}")
    print(f"wrote difficulty readout to {result.readout_path}")
    if args.verbose:
        ok_rows = sum(1 for row in result.rows if row.get("status") == "ok")
        failed_rows = len(result.rows) - ok_rows
        print(f"rows={len(result.rows)} ok={ok_rows} non_ok={failed_rows} candidate_rows={len(result.candidate_rows)}")


def _config_payload(config: CausalOPEBenchmarkConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir())
    return payload


def _calibration_config_payload(config: CalibrationStudyConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir())
    return payload


def _stress_config_payload(config: DifficultyStressStudyConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir())
    return payload


def _validate_cli_config(config: CausalOPEBenchmarkConfig, *, target_policies_supplied: bool) -> None:
    known_estimators = {estimator.name for estimator in list_estimators()}
    unknown = sorted(set(str(estimator) for estimator in config.estimators) - known_estimators)
    if unknown:
        raise ConfigurationError(f"Unknown estimator(s): {', '.join(unknown)}. Valid estimators: {', '.join(sorted(known_estimators))}.")
    _validate_target_policies(tuple(config.families), tuple(config.target_policies), target_policies_supplied=target_policies_supplied)


def _validate_target_policies(families: tuple[str, ...], target_policies: tuple[str, ...], *, target_policies_supplied: bool) -> None:
    for family in families:
        if family == "streamlift" and not target_policies_supplied:
            continue
        allowed = POLICY_NAMES_BY_FAMILY.get(family, ())
        invalid = sorted(set(target_policies) - set(allowed))
        if invalid:
            raise ConfigurationError(
                f"Unknown target policy for {family}: {', '.join(invalid)}. Valid policies: {', '.join(allowed)}."
            )


def _exit_with_error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
