from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from causal_ope_benchmark.config import DomainScenario, FamilyName
from causal_ope_benchmark.exceptions import ConfigurationError


DifficultyName = Literal["easy", "medium", "hard", "realistic"]
StressScale = Literal["ci", "audit", "exhaustive"]

DIFFICULTY_NAMES: tuple[DifficultyName, ...] = ("easy", "medium", "hard", "realistic")
STRESS_SCALES: tuple[StressScale, ...] = ("ci", "audit", "exhaustive")


@dataclass(frozen=True)
class DifficultySpec:
    """Public, reproducible description of one statistical difficulty level.

    Difficulty is intentionally separated from runtime scale. The profiles
    below keep primary cells identifiable and time-invariant: hard settings are
    hard because of overlap, sample size, nonlinear dynamics, delayed effects,
    action constraints, dose/cost mechanisms, and observed missingness/censoring,
    not because the estimand is unidentified.
    """

    name: DifficultyName
    summary: str
    overlap: str
    target_policy_distance_range: tuple[float, float]
    confounding: str
    delayed_effects: str
    missingness: str
    censoring: str
    action_constraints: bool
    subgroup_heterogeneity: float
    streamlift_observed_horizons: tuple[int, ...]
    streamlift_include_infinite_horizon: bool
    trajectory_horizon: int
    gammas: tuple[float, ...]
    target_policies: tuple[str, ...]
    sample_sizes: tuple[tuple[FamilyName, int], ...]

    def sample_size_for_family(self, family: FamilyName) -> int:
        """Return the representative sample size for a family."""

        lookup = dict(self.sample_sizes)
        return int(lookup.get(family, lookup.get("streamretain", 1000)))


@dataclass(frozen=True)
class DifficultyCell:
    """One generated scenario cell used by difficulty stress studies."""

    difficulty: DifficultyName
    stress_dimension: str
    scenario: DomainScenario
    primary: bool = True
    sensitivity: bool = False


_SPECS: dict[str, DifficultySpec] = {
    "easy": DifficultySpec(
        name="easy",
        summary="Generous support, randomized or lightly adjusted data, short horizons, and mild nonlinearities.",
        overlap="good",
        target_policy_distance_range=(0.15, 0.25),
        confounding="randomized",
        delayed_effects="mild",
        missingness="none",
        censoring="administrative",
        action_constraints=False,
        subgroup_heterogeneity=0.20,
        streamlift_observed_horizons=(3,),
        streamlift_include_infinite_horizon=False,
        trajectory_horizon=18,
        gammas=(0.90,),
        target_policies=("moderate",),
        sample_sizes=(("streamlift", 600), ("streamretain", 1200), ("clinic_dtr", 1200), ("epicare", 800)),
    ),
    "medium": DifficultySpec(
        name="medium",
        summary="Moderate policy shift and observed confounding with all identifying covariates public.",
        overlap="moderate",
        target_policy_distance_range=(0.40, 0.55),
        confounding="observed",
        delayed_effects="moderate",
        missingness="mcar",
        censoring="administrative",
        action_constraints=True,
        subgroup_heterogeneity=0.55,
        streamlift_observed_horizons=(2,),
        streamlift_include_infinite_horizon=False,
        trajectory_horizon=24,
        gammas=(0.95,),
        target_policies=("conservative", "moderate"),
        sample_sizes=(("streamlift", 1000), ("streamretain", 1000), ("clinic_dtr", 1000), ("epicare", 800)),
    ),
    "hard": DifficultySpec(
        name="hard",
        summary="Weak but nonzero overlap, stronger nonlinear/delayed mechanisms, smaller samples, and observed MAR/censoring challenges.",
        overlap="weak",
        target_policy_distance_range=(0.70, 0.85),
        confounding="observed",
        delayed_effects="strong",
        missingness="mar",
        censoring="informative",
        action_constraints=True,
        subgroup_heterogeneity=1.00,
        streamlift_observed_horizons=(1, 2),
        streamlift_include_infinite_horizon=True,
        trajectory_horizon=36,
        gammas=(0.97,),
        target_policies=("moderate", "aggressive", "safety_constrained"),
        sample_sizes=(("streamlift", 1200), ("streamretain", 700), ("clinic_dtr", 700), ("epicare", 600)),
    ),
    "realistic": DifficultySpec(
        name="realistic",
        summary="Deployment-like mix of moderate support, public observed confounding, constraints, dose/cost effects, and plausible missingness.",
        overlap="moderate",
        target_policy_distance_range=(0.45, 0.65),
        confounding="observed",
        delayed_effects="realistic",
        missingness="mar",
        censoring="administrative",
        action_constraints=True,
        subgroup_heterogeneity=0.75,
        streamlift_observed_horizons=(2, 3),
        streamlift_include_infinite_horizon=True,
        trajectory_horizon=24,
        gammas=(0.95,),
        target_policies=("conservative", "moderate", "aggressive", "budget_constrained", "safety_constrained"),
        sample_sizes=(("streamlift", 1500), ("streamretain", 1500), ("clinic_dtr", 1200), ("epicare", 1000)),
    ),
}


def list_difficulties() -> tuple[DifficultySpec, ...]:
    """List registered user-facing difficulty profiles."""

    return tuple(_SPECS[name] for name in DIFFICULTY_NAMES)


def describe_difficulty(name: str) -> DifficultySpec:
    """Return a registered difficulty profile."""

    key = _normalize_difficulty(name)
    return _SPECS[key]


def difficulty_cells(
    difficulty: str,
    family: FamilyName,
    *,
    scale: str = "audit",
    include_sensitivity: bool = False,
) -> tuple[DifficultyCell, ...]:
    """Return stationary, identifiable scenario cells for a difficulty study."""

    key = _normalize_difficulty(difficulty)
    fam = _normalize_family(family)
    stress_scale = _normalize_scale(scale)
    base = _base_cells(_SPECS[key], fam)
    if stress_scale == "ci":
        return base[:1]
    cells = list(base)
    cells.extend(_one_factor_cells(_SPECS[key], fam))
    if stress_scale == "exhaustive":
        cells.extend(_exhaustive_interaction_cells(_SPECS[key], fam))
    if include_sensitivity:
        cells.extend(_sensitivity_cells(_SPECS[key], fam))
    return tuple(cells)


def scenarios_for_difficulty(
    difficulty: str,
    family: FamilyName,
    *,
    scale: str = "audit",
    include_sensitivity: bool = False,
) -> tuple[DomainScenario, ...]:
    """Return public DomainScenario objects for a difficulty profile."""

    return tuple(cell.scenario for cell in difficulty_cells(difficulty, family, scale=scale, include_sensitivity=include_sensitivity))


def sample_sizes_for_difficulty(difficulty: str, family: FamilyName, *, scale: str = "audit") -> tuple[int, ...]:
    """Return scale-adjusted sample-size ladder for a difficulty/family pair."""

    spec = describe_difficulty(difficulty)
    base = spec.sample_size_for_family(_normalize_family(family))
    stress_scale = _normalize_scale(scale)
    if stress_scale == "ci":
        smoke_cap = 1000 if str(family) == "streamlift" else 200
        return (max(18, min(base, smoke_cap)),)
    if stress_scale == "audit":
        return (base,)
    return (max(50, base // 2), base, base * 2)


def seeds_for_scale(scale: str) -> tuple[int, ...]:
    """Return deterministic seed sets for a stress-study scale."""

    stress_scale = _normalize_scale(scale)
    if stress_scale == "ci":
        return (0,)
    if stress_scale == "audit":
        return (0, 1, 2)
    return tuple(range(10))


def target_policies_for_difficulty(difficulty: str, family: FamilyName) -> tuple[str, ...]:
    """Return target policies requested by a difficulty, filtered by family."""

    from causal_ope_benchmark.policies import POLICY_NAMES_BY_FAMILY

    spec = describe_difficulty(difficulty)
    allowed = set(POLICY_NAMES_BY_FAMILY.get(_normalize_family(family), ()))
    return tuple(policy for policy in spec.target_policies if policy in allowed)


def _base_cells(spec: DifficultySpec, family: FamilyName) -> tuple[DifficultyCell, ...]:
    low, high = _target_distance_range(spec, family)
    mid = 0.5 * (low + high)
    delay = _delay_pattern(spec, family)
    common = dict(
        nonstationarity=False,
        action_constraints=bool(spec.action_constraints),
        subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
        surrogate_validity="valid",
        streamlift_campaign_mode="finite_campaign",
        campaign_length=3,
        leaderboard_eligible=True,
    )
    cells = (
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="base_clean",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_clean_stationary_good_overlap",
                overlap="good",
                confounding="randomized",
                delay_pattern=delay,
                missingness="none",
                censoring="administrative",
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=True),
                target_policy_distance=low,
                **common,
            ),
            primary=True,
        ),
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="base_observed",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_observed_stationary_{spec.overlap}_overlap",
                overlap=spec.overlap,
                confounding="observed",
                delay_pattern=delay,
                missingness=spec.missingness,
                censoring=spec.censoring,
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                target_policy_distance=mid,
                **common,
            ),
            primary=True,
        ),
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="base_domain_realistic",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_domain_realistic_stationary",
                overlap="moderate" if spec.name in {"easy", "realistic"} else spec.overlap,
                confounding="observed",
                delay_pattern=delay,
                missingness=spec.missingness,
                censoring=spec.censoring,
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                action_constraints=True,
                subgroup_heterogeneity=max(float(spec.subgroup_heterogeneity), 0.5),
                target_policy_distance=high,
                surrogate_validity="valid",
                nonstationarity=False,
                streamlift_campaign_mode="finite_campaign",
                campaign_length=3,
                leaderboard_eligible=True,
            ),
            primary=True,
        ),
    )
    return cells


def _one_factor_cells(spec: DifficultySpec, family: FamilyName) -> tuple[DifficultyCell, ...]:
    delay = _delay_pattern(spec, family)
    _, high = _target_distance_range(spec, family)
    out: list[DifficultyCell] = []
    for value in (0.15, 0.35, 0.55, 0.75, 0.90):
        out.append(
            DifficultyCell(
                difficulty=spec.name,
                stress_dimension="target_policy_distance",
                scenario=DomainScenario(
                    name=f"difficulty_{spec.name}_target_distance_{str(value).replace('.', '_')}",
                    overlap=spec.overlap if value >= 0.55 else "moderate",
                    confounding="observed",
                    delay_pattern=delay,
                    missingness=spec.missingness,
                    censoring=spec.censoring,
                    noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                    action_constraints=True,
                    subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                    target_policy_distance=value,
                    nonstationarity=False,
                    leaderboard_eligible=True,
                ),
                primary=value <= 0.90,
            )
        )
    for overlap in ("good", "moderate", "weak"):
        out.append(
            DifficultyCell(
                difficulty=spec.name,
                stress_dimension="overlap",
                scenario=DomainScenario(
                    name=f"difficulty_{spec.name}_overlap_{overlap}",
                    overlap=overlap,
                    confounding="observed",
                    delay_pattern=delay,
                    missingness=spec.missingness,
                    censoring=spec.censoring,
                    noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                    action_constraints=True,
                    subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                    target_policy_distance=min(0.85, high),
                    nonstationarity=False,
                    leaderboard_eligible=True,
                ),
                primary=True,
            )
        )
    out.extend(
        [
            DifficultyCell(
                difficulty=spec.name,
                stress_dimension="missingness",
                scenario=DomainScenario(
                    name=f"difficulty_{spec.name}_mar_missingness",
                    overlap=spec.overlap,
                    confounding="observed",
                    delay_pattern=delay,
                    missingness="mar",
                    censoring=spec.censoring,
                    noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                    action_constraints=True,
                    subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                    target_policy_distance=high,
                    nonstationarity=False,
                    leaderboard_eligible=True,
                ),
                primary=True,
            ),
            DifficultyCell(
                difficulty=spec.name,
                stress_dimension="censoring",
                scenario=DomainScenario(
                    name=f"difficulty_{spec.name}_observed_informative_censoring",
                    overlap=spec.overlap,
                    confounding="observed",
                    delay_pattern=delay,
                    missingness=spec.missingness,
                    censoring="informative",
                    noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                    action_constraints=True,
                    subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                    target_policy_distance=high,
                    nonstationarity=False,
                    leaderboard_eligible=True,
                ),
                primary=True,
            ),
        ]
    )
    if family == "streamlift":
        out.append(
            DifficultyCell(
                difficulty=spec.name,
                stress_dimension="noncompliance",
                scenario=DomainScenario(
                    name=f"difficulty_{spec.name}_streamlift_noncompliance",
                    overlap=spec.overlap,
                    confounding="observed",
                    delay_pattern=delay,
                    missingness=spec.missingness,
                    censoring=spec.censoring,
                    noncompliance_rate=0.18 if spec.name in {"hard", "realistic"} else 0.10,
                    action_constraints=True,
                    subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                    target_policy_distance=high,
                    nonstationarity=False,
                    leaderboard_eligible=True,
                ),
                primary=True,
            )
        )
    return tuple(out)


def _exhaustive_interaction_cells(spec: DifficultySpec, family: FamilyName) -> tuple[DifficultyCell, ...]:
    delay = _delay_pattern(spec, family)
    _, high = _target_distance_range(spec, family)
    return (
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="weak_overlap_long_horizon",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_weak_overlap_long_horizon",
                overlap="weak",
                confounding="observed",
                delay_pattern=delay,
                missingness=spec.missingness,
                censoring=spec.censoring,
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                action_constraints=True,
                subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                target_policy_distance=min(0.90, high + 0.05),
                nonstationarity=False,
                leaderboard_eligible=True,
            ),
            primary=True,
        ),
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="observed_confounding_delayed_effects",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_observed_delayed_nonlinear",
                overlap=spec.overlap,
                confounding="observed",
                delay_pattern="short_harm_long_benefit" if family == "streamlift" else "delayed_benefit",
                missingness=spec.missingness,
                censoring=spec.censoring,
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                action_constraints=True,
                subgroup_heterogeneity=max(float(spec.subgroup_heterogeneity), 0.8),
                target_policy_distance=high,
                nonstationarity=False,
                leaderboard_eligible=True,
            ),
            primary=True,
        ),
    )


def _sensitivity_cells(spec: DifficultySpec, family: FamilyName) -> tuple[DifficultyCell, ...]:
    """Assumption-violation cells excluded from difficulty verdicts."""

    _, high = _target_distance_range(spec, family)
    return (
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="assumption_violation_latent_confounding",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_latent_confounding_sensitivity",
                overlap="moderate",
                confounding="latent",
                delay_pattern=_delay_pattern(spec, family),
                missingness="informative",
                censoring="informative",
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                action_constraints=True,
                subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                target_policy_distance=min(1.0, high + 0.10),
                nonstationarity=False,
                leaderboard_eligible=False,
            ),
            primary=False,
            sensitivity=True,
        ),
        DifficultyCell(
            difficulty=spec.name,
            stress_dimension="assumption_violation_nonstationarity",
            scenario=DomainScenario(
                name=f"difficulty_{spec.name}_regime_shift_sensitivity",
                overlap=spec.overlap,
                confounding="observed",
                delay_pattern=_delay_pattern(spec, family),
                missingness=spec.missingness,
                censoring=spec.censoring,
                noncompliance_rate=_streamlift_noncompliance(spec.name, family, mild=False),
                action_constraints=True,
                subgroup_heterogeneity=float(spec.subgroup_heterogeneity),
                target_policy_distance=high,
                nonstationarity=True,
                leaderboard_eligible=False,
            ),
            primary=False,
            sensitivity=True,
        ),
    )


def _delay_pattern(spec: DifficultySpec, family: FamilyName) -> str:
    if family == "streamlift":
        if spec.name == "easy":
            return "immediate"
        if spec.name in {"hard", "realistic"}:
            return "short_harm_long_benefit"
        return "delayed_benefit"
    if spec.name == "easy":
        return "immediate"
    return "delayed_benefit"


def _target_distance_range(spec: DifficultySpec, family: FamilyName) -> tuple[float, float]:
    if family != "streamlift":
        return spec.target_policy_distance_range
    if spec.name == "easy":
        return (0.10, 0.20)
    if spec.name == "medium":
        return (0.15, 0.30)
    if spec.name == "hard":
        return (0.30, 0.55)
    return (0.20, 0.40)


def _streamlift_noncompliance(spec_name: str, family: FamilyName, *, mild: bool) -> float:
    if family != "streamlift":
        return 0.0
    if mild:
        return 0.03 if spec_name == "easy" else 0.06
    return {"easy": 0.05, "medium": 0.10, "hard": 0.16, "realistic": 0.12}[spec_name]


def _normalize_difficulty(name: str) -> DifficultyName:
    key = str(name).lower().replace("_", "-")
    aliases = {"core": "medium", "core-lite": "medium"}
    key = aliases.get(key, key)
    if key not in DIFFICULTY_NAMES:
        raise ConfigurationError(f"Unknown difficulty {name!r}. Valid difficulties: {', '.join(DIFFICULTY_NAMES)}.")
    return key  # type: ignore[return-value]


def _normalize_scale(scale: str) -> StressScale:
    key = str(scale).lower()
    if key not in STRESS_SCALES:
        raise ConfigurationError(f"Unknown stress-study scale {scale!r}. Valid scales: {', '.join(STRESS_SCALES)}.")
    return key  # type: ignore[return-value]


def _normalize_family(family: str) -> FamilyName:
    key = str(family)
    if key not in {"streamlift", "streamretain", "clinic_dtr", "epicare"}:
        raise ConfigurationError("Unknown family {!r}. Valid families: streamlift, streamretain, clinic_dtr, epicare.".format(family))
    return key  # type: ignore[return-value]


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in items))
