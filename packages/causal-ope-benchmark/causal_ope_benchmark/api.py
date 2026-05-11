from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from causal_ope_benchmark.calibration import CalibrationRunResult, CalibrationStudyConfig, run_calibration_study
from causal_ope_benchmark.config import CausalOPEBenchmarkConfig, DomainScenario, scenarios_for_profile
from causal_ope_benchmark.constants import ALL_FAMILIES, DEFAULT_FAMILIES, DEFAULT_OUTPUT_FILES, DIFFICULTY_OUTPUT_FILES, PACKAGE_VERSION
from causal_ope_benchmark.difficulty import DifficultySpec, describe_difficulty, list_difficulties, scenarios_for_difficulty
from causal_ope_benchmark.exceptions import ConfigurationError
from causal_ope_benchmark.io import read_csv, read_json
from causal_ope_benchmark.policies import ACTION_NAMES_BY_FAMILY, POLICY_NAMES_BY_FAMILY
from causal_ope_benchmark.runner import BenchmarkRunResult, make_problem, run_benchmark
from causal_ope_benchmark.stress import DifficultyStressRunResult, DifficultyStressStudyConfig, run_difficulty_stress_study
from causal_ope_benchmark.types import BenchmarkProblem
from causal_ope_benchmark.validation import validate_calibration_output_bundle, validate_difficulty_output_bundle, validate_output_bundle, validate_problem

__all__ = [
    "CalibrationOutputBundle",
    "CalibrationRunResult",
    "CalibrationStudyConfig",
    "DifficultyOutputBundle",
    "DifficultySpec",
    "DifficultyStressRunResult",
    "DifficultyStressStudyConfig",
    "EstimatorInfo",
    "FamilyInfo",
    "OutputBundle",
    "describe_difficulty",
    "describe_family",
    "list_estimators",
    "list_difficulties",
    "list_families",
    "list_target_policies",
    "load_calibration_results",
    "load_difficulty_results",
    "load_results",
    "make_benchmark_problem",
    "package_version",
    "run_calibration",
    "run_calibration_study",
    "run_difficulty_stress_study",
    "run_difficulty_study",
    "run_suite",
    "validate_calibration_output_bundle",
    "validate_difficulty_output_bundle",
    "validate_output_bundle",
    "validate_problem",
]


@dataclass(frozen=True)
class FamilyInfo:
    """Public descriptor for a benchmark family."""

    name: str
    display_name: str
    summary: str
    default_profile_member: bool
    native: bool
    external: bool
    gym_env_available: bool
    scope_rl_export_available: bool
    target_policies: tuple[str, ...]
    action_names: tuple[str, ...]


@dataclass(frozen=True)
class EstimatorInfo:
    """Public descriptor for an estimator shipped with the benchmark."""

    name: str
    summary: str
    families: tuple[str, ...]
    diagnostic_only: bool = False
    optional_dependency: str | None = None


@dataclass(frozen=True)
class OutputBundle:
    """Loaded benchmark output files."""

    output_dir: Path
    results_path: Path
    summary_path: Path
    tuning_path: Path
    diagnostics_path: Path
    manifest_path: Path
    readout_path: Path
    output_schema_path: Path
    results: list[dict[str, str]]
    summary: list[dict[str, str]]
    tuning_results: list[dict[str, str]]
    diagnostics: dict[str, Any]
    manifest: dict[str, Any]
    output_schema: dict[str, Any]


@dataclass(frozen=True)
class CalibrationOutputBundle:
    """Loaded calibration-study output files."""

    output_dir: Path
    results_path: Path
    summary_path: Path
    candidates_path: Path
    manifest_path: Path
    readout_path: Path
    results: list[dict[str, str]]
    summary: list[dict[str, str]]
    candidates: list[dict[str, str]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DifficultyOutputBundle:
    """Loaded difficulty stress-study output files."""

    output_dir: Path
    results_path: Path
    summary_path: Path
    candidates_path: Path
    manifest_path: Path
    readout_path: Path
    results: list[dict[str, str]]
    summary: list[dict[str, str]]
    candidates: list[dict[str, str]]
    manifest: dict[str, Any]


_FAMILY_SUMMARIES = {
    "streamlift": "Short-panel A/B or observed-confounded experiments for long-horizon retention and revenue effects.",
    "streamretain": "Streaming subscription lifecycle OPE with retention, revenue, contact, spend, and fatigue constraints.",
    "clinic_dtr": "Transparent cardiometabolic dynamic treatment-regime OPE with survival, biomarkers, dose, and safety.",
    "epicare": "Optional external EpiCare healthcare RL benchmark adapter loaded through Gym.",
}


_ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo("naive_short_term", "Short-term StreamLift extrapolation baseline.", ("streamlift",)),
    EstimatorInfo(
        "streamlift_stratified_gcomp",
        "StreamLift arm-stratified stationary dynamics g-computation baseline.",
        ("streamlift",),
    ),
    EstimatorInfo("direct_method", "Ridge direct-method baseline using exact discrete target-policy expectations.", ALL_FAMILIES),
    EstimatorInfo("ipw", "Trajectory importance weighting using pi_e(A|S) over behavior propensities.", ALL_FAMILIES),
    EstimatorInfo("snipw", "Self-normalized IPW baseline.", ALL_FAMILIES),
    EstimatorInfo("doubly_robust", "Simple direct-plus-IPW doubly robust baseline.", ALL_FAMILIES),
    EstimatorInfo("linear_fqe", "Cheap linear FQE diagnostic for sequential families.", ("streamretain", "clinic_dtr", "epicare")),
    EstimatorInfo("boosted_fqe", "Package-integrated LightGBM FQE.", ("streamretain", "clinic_dtr", "epicare"), optional_dependency="fqe/lightgbm"),
    EstimatorInfo("boosted_fqe_auto", "Package-integrated LightGBM FQE with proxy AutoML.", ("streamretain", "clinic_dtr", "epicare"), optional_dependency="fqe/lightgbm"),
    EstimatorInfo("neural_fqe", "Package-integrated neural FQE.", ("streamretain", "clinic_dtr", "epicare"), optional_dependency="fqe/torch"),
    EstimatorInfo("neural_fqe_auto", "Package-integrated neural FQE with proxy AutoML.", ("streamretain", "clinic_dtr", "epicare"), optional_dependency="fqe/torch"),
    EstimatorInfo(
        "discounted_occupancy_boosted",
        "Package-integrated boosted discounted-occupancy weighted OPE.",
        ("streamretain", "clinic_dtr", "epicare"),
        optional_dependency="occupancy-ratio/lightgbm",
    ),
    EstimatorInfo(
        "discounted_occupancy_neural",
        "Package-integrated neural discounted-occupancy weighted OPE.",
        ("streamretain", "clinic_dtr", "epicare"),
        optional_dependency="occupancy-ratio/torch",
    ),
    EstimatorInfo(
        "discounted_occupancy_boosted_auto",
        "Boosted discounted-occupancy OPE with proxy AutoML.",
        ("streamretain", "clinic_dtr", "epicare"),
        optional_dependency="occupancy-ratio/lightgbm",
    ),
    EstimatorInfo(
        "discounted_occupancy_neural_auto",
        "Neural discounted-occupancy OPE with proxy AutoML.",
        ("streamretain", "clinic_dtr", "epicare"),
        optional_dependency="occupancy-ratio/torch",
    ),
    EstimatorInfo("oracle_selected_fqe_diagnostic", "Diagnostic-only sealed-truth FQE family selector.", ("streamretain", "clinic_dtr", "epicare"), diagnostic_only=True),
    EstimatorInfo(
        "oracle_selected_discounted_occupancy_diagnostic",
        "Diagnostic-only sealed-truth discounted-occupancy family selector.",
        ("streamretain", "clinic_dtr", "epicare"),
        diagnostic_only=True,
    ),
    EstimatorInfo("neural_occupancy", "Calibration-study neural discounted occupancy-ratio OPE.", ALL_FAMILIES, optional_dependency="occupancy-ratio/torch"),
    EstimatorInfo(
        "neural_fqe_streamlift_diagnostic",
        "Diagnostic-only neural FQE row for StreamLift transition sanity checks.",
        ("streamlift",),
        diagnostic_only=True,
        optional_dependency="fqe/torch",
    ),
    EstimatorInfo("ipcw_rmst", "IPCW/RMST survival baseline for ClinicDTR.", ("clinic_dtr",)),
    EstimatorInfo("oracle_diagnostic", "Scorer-only diagnostic upper-bound row.", ALL_FAMILIES, diagnostic_only=True),
)


def list_families(*, include_external: bool = True) -> tuple[FamilyInfo, ...]:
    """List registered benchmark families."""

    names = ("streamlift", "streamretain", "clinic_dtr", "epicare") if include_external else DEFAULT_FAMILIES
    return tuple(describe_family(name) for name in names)


def describe_family(family: str) -> FamilyInfo:
    """Return a public descriptor for one benchmark family."""

    key = _normalize_family(family)
    action_names = ACTION_NAMES_BY_FAMILY.get(key, tuple(f"action_{i}" for i in range(2)) if key == "epicare" else ())
    return FamilyInfo(
        name=key,
        display_name={"streamlift": "StreamLift", "streamretain": "StreamRetain", "clinic_dtr": "ClinicDTR", "epicare": "EpiCare"}[key],
        summary=_FAMILY_SUMMARIES[key],
        default_profile_member=key in DEFAULT_FAMILIES,
        native=key in DEFAULT_FAMILIES,
        external=key == "epicare",
        gym_env_available=key in {"streamretain", "clinic_dtr", "epicare"},
        scope_rl_export_available=True,
        target_policies=POLICY_NAMES_BY_FAMILY.get(key, ()),
        action_names=action_names,
    )


def list_target_policies(family: str) -> tuple[str, ...]:
    """List named target policies for a family."""

    return describe_family(family).target_policies


def list_estimators() -> tuple[EstimatorInfo, ...]:
    """List estimators known to the default runner."""

    return _ESTIMATORS


def make_benchmark_problem(
    family: str,
    *,
    profile: str = "smoke",
    difficulty: str | None = None,
    scenario: DomainScenario | None = None,
    sample_size: int = 120,
    gamma: float = 0.90,
    seed: int = 0,
    observed_horizon: int = 2,
    target_policy: str = "moderate",
    config: CausalOPEBenchmarkConfig | None = None,
) -> BenchmarkProblem:
    """Create one benchmark problem through the stable public facade."""

    key = _normalize_family(family)
    cfg = config or CausalOPEBenchmarkConfig.for_profile(profile)
    if scenario is not None:
        cell = scenario
    elif difficulty is not None:
        cell = scenarios_for_difficulty(difficulty, key, scale="ci")[0]  # type: ignore[arg-type]
    else:
        cell = scenarios_for_profile(cfg.profile, key)[0]
    return make_problem(
        family=key,
        scenario=cell,
        sample_size=int(sample_size),
        gamma=float(gamma),
        seed=int(seed),
        observed_horizon=int(observed_horizon),
        target_policy=str(target_policy),
        config=cfg,
    )


def run_suite(config: CausalOPEBenchmarkConfig | None = None, **overrides: Any) -> BenchmarkRunResult:
    """Run the benchmark suite with optional config overrides."""

    base = config or CausalOPEBenchmarkConfig.for_profile(str(overrides.pop("profile", "smoke")))
    if overrides:
        base = CausalOPEBenchmarkConfig(**{**base.__dict__, **overrides})
    return run_benchmark(base)


def run_calibration(config: CalibrationStudyConfig | None = None, **overrides: Any) -> CalibrationRunResult:
    """Run the neural FQE and occupancy-ratio calibration study."""

    return run_calibration_study(config, **overrides)


def run_difficulty_study(
    config: DifficultyStressStudyConfig | None = None,
    **overrides: Any,
) -> DifficultyStressRunResult:
    """Run systematic difficulty stress tests."""

    return run_difficulty_stress_study(config, **overrides)


def load_results(output_dir: str | Path) -> OutputBundle:
    """Load a completed benchmark output directory."""

    root = Path(output_dir)
    paths = {key: root / filename for key, filename in DEFAULT_OUTPUT_FILES.items()}
    return OutputBundle(
        output_dir=root,
        results_path=paths["results"],
        summary_path=paths["summary"],
        tuning_path=paths["tuning_results"],
        diagnostics_path=paths["diagnostics"],
        manifest_path=paths["manifest"],
        readout_path=paths["readout"],
        output_schema_path=paths["output_schema"],
        results=read_csv(paths["results"]) if paths["results"].exists() else [],
        summary=read_csv(paths["summary"]) if paths["summary"].exists() else [],
        tuning_results=read_csv(paths["tuning_results"]) if paths["tuning_results"].exists() else [],
        diagnostics=read_json(paths["diagnostics"]) if paths["diagnostics"].exists() else {},
        manifest=read_json(paths["manifest"]) if paths["manifest"].exists() else {},
        output_schema=read_json(paths["output_schema"]) if paths["output_schema"].exists() else {},
    )


def load_calibration_results(output_dir: str | Path) -> CalibrationOutputBundle:
    """Load a completed calibration output directory."""

    root = Path(output_dir)
    return CalibrationOutputBundle(
        output_dir=root,
        results_path=root / "calibration_results.csv",
        summary_path=root / "calibration_summary.csv",
        candidates_path=root / "calibration_candidates.csv",
        manifest_path=root / "calibration_manifest.json",
        readout_path=root / "calibration_readout.md",
        results=read_csv(root / "calibration_results.csv") if (root / "calibration_results.csv").exists() else [],
        summary=read_csv(root / "calibration_summary.csv") if (root / "calibration_summary.csv").exists() else [],
        candidates=read_csv(root / "calibration_candidates.csv") if (root / "calibration_candidates.csv").exists() else [],
        manifest=read_json(root / "calibration_manifest.json") if (root / "calibration_manifest.json").exists() else {},
    )


def load_difficulty_results(output_dir: str | Path) -> DifficultyOutputBundle:
    """Load a completed difficulty stress-study output directory."""

    root = Path(output_dir)
    return DifficultyOutputBundle(
        output_dir=root,
        results_path=root / DIFFICULTY_OUTPUT_FILES["results"],
        summary_path=root / DIFFICULTY_OUTPUT_FILES["summary"],
        candidates_path=root / DIFFICULTY_OUTPUT_FILES["candidates"],
        manifest_path=root / DIFFICULTY_OUTPUT_FILES["manifest"],
        readout_path=root / DIFFICULTY_OUTPUT_FILES["readout"],
        results=read_csv(root / DIFFICULTY_OUTPUT_FILES["results"]) if (root / DIFFICULTY_OUTPUT_FILES["results"]).exists() else [],
        summary=read_csv(root / DIFFICULTY_OUTPUT_FILES["summary"]) if (root / DIFFICULTY_OUTPUT_FILES["summary"]).exists() else [],
        candidates=read_csv(root / DIFFICULTY_OUTPUT_FILES["candidates"]) if (root / DIFFICULTY_OUTPUT_FILES["candidates"]).exists() else [],
        manifest=read_json(root / DIFFICULTY_OUTPUT_FILES["manifest"]) if (root / DIFFICULTY_OUTPUT_FILES["manifest"]).exists() else {},
    )


def package_version() -> str:
    """Return the benchmark package version."""

    return PACKAGE_VERSION


def _normalize_family(family: str) -> str:
    key = str(family)
    if key not in {"streamlift", "streamretain", "clinic_dtr", "epicare"}:
        choices = ", ".join(["streamlift", "streamretain", "clinic_dtr", "epicare"])
        raise ConfigurationError(f"Unknown family {key!r}. Valid families: {choices}.")
    return key
