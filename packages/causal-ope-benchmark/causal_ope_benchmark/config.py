from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


Profile = Literal["smoke", "core", "full", "paper"]
FamilyName = Literal["streamlift", "streamretain", "clinic_dtr", "epicare"]
AutoMLTuning = Literal["off", "fast", "balanced"]
OverlapLevel = Literal["good", "moderate", "weak", "structural_gap"]
ConfoundingLevel = Literal["randomized", "observed", "latent"]
DelayPattern = Literal["immediate", "delayed_benefit", "short_harm_long_benefit", "short_benefit_long_harm"]
MissingnessPattern = Literal["none", "mcar", "mar", "informative"]
CensoringPattern = Literal["none", "administrative", "informative", "competing_risk"]
SurrogateValidity = Literal["valid", "weak", "misleading", "sign_reversal"]
StreamLiftCampaignMode = Literal["one_shot", "finite_campaign", "persistent", "finite", "always_on"]


@dataclass(frozen=True)
class DomainScenario:
    """Scenario knobs shared by realistic benchmark families."""

    name: str
    overlap: OverlapLevel = "good"
    confounding: ConfoundingLevel = "randomized"
    delay_pattern: DelayPattern = "immediate"
    nonstationarity: bool = False
    missingness: MissingnessPattern = "none"
    censoring: CensoringPattern = "administrative"
    noncompliance_rate: float = 0.0
    action_constraints: bool = True
    subgroup_heterogeneity: float = 0.5
    target_policy_distance: float = 0.6
    surrogate_validity: SurrogateValidity = "valid"
    streamlift_campaign_mode: StreamLiftCampaignMode = "finite_campaign"
    campaign_length: int = 3
    leaderboard_eligible: bool = True

    def __post_init__(self) -> None:
        if self.overlap not in {"good", "moderate", "weak", "structural_gap"}:
            raise ValueError("overlap has an unsupported value.")
        if self.confounding not in {"randomized", "observed", "latent"}:
            raise ValueError("confounding has an unsupported value.")
        if self.delay_pattern not in {
            "immediate",
            "delayed_benefit",
            "short_harm_long_benefit",
            "short_benefit_long_harm",
        }:
            raise ValueError("delay_pattern has an unsupported value.")
        if self.missingness not in {"none", "mcar", "mar", "informative"}:
            raise ValueError("missingness has an unsupported value.")
        if self.censoring not in {"none", "administrative", "informative", "competing_risk"}:
            raise ValueError("censoring has an unsupported value.")
        if self.surrogate_validity not in {"valid", "weak", "misleading", "sign_reversal"}:
            raise ValueError("surrogate_validity has an unsupported value.")
        if not (0.0 <= float(self.noncompliance_rate) <= 1.0):
            raise ValueError("noncompliance_rate must be in [0, 1].")
        if float(self.subgroup_heterogeneity) < 0.0:
            raise ValueError("subgroup_heterogeneity must be nonnegative.")
        if float(self.target_policy_distance) < 0.0:
            raise ValueError("target_policy_distance must be nonnegative.")
        if float(self.target_policy_distance) > 1.0:
            raise ValueError("target_policy_distance must be in [0, 1].")
        if self.streamlift_campaign_mode == "finite":
            object.__setattr__(self, "streamlift_campaign_mode", "finite_campaign")
        elif self.streamlift_campaign_mode == "always_on":
            object.__setattr__(self, "streamlift_campaign_mode", "persistent")
        if self.streamlift_campaign_mode not in {"one_shot", "finite_campaign", "persistent"}:
            raise ValueError("streamlift_campaign_mode has an unsupported value.")
        if int(self.campaign_length) <= 0:
            raise ValueError("campaign_length must be positive.")


@dataclass(frozen=True)
class CausalOPEBenchmarkConfig:
    """User-facing configuration for the causal OPE benchmark runner."""

    profile: Profile = "smoke"
    output_root: Path = Path("outputs/causal_ope_benchmark")
    seeds: Sequence[int] = field(default_factory=lambda: (0,))
    families: Sequence[FamilyName] = field(default_factory=lambda: ("streamlift", "streamretain", "clinic_dtr"))
    sample_sizes: Sequence[int] = field(default_factory=lambda: (120,))
    gammas: Sequence[float] = field(default_factory=lambda: (0.95,))
    observed_horizons: Sequence[int] = field(default_factory=lambda: (3,))
    target_policies: Sequence[str] = field(default_factory=lambda: ("moderate",))
    estimators: Sequence[str] = field(
        default_factory=lambda: (
            "naive_short_term",
            "streamlift_stratified_gcomp",
            "direct_method",
            "ipw",
            "snipw",
            "doubly_robust",
            "linear_fqe",
            "boosted_fqe",
            "neural_fqe",
            "neural_fqe_streamlift_diagnostic",
            "ipcw_rmst",
            "oracle_diagnostic",
        )
    )
    forecast_horizons: Sequence[int] = field(default_factory=lambda: (6, 12, 24, 36))
    trajectory_horizon: int = 24
    streamlift_long_horizon: int = 36
    streamlift_include_infinite_horizon: bool = False
    streamlift_infinite_horizon_max_steps: int = 240
    mc_truth_rollouts: int = 96
    fqe_hidden_dims: Sequence[int] = field(default_factory=lambda: (32, 32))
    fqe_num_iterations: int = 24
    fqe_gradient_steps_per_iteration: int = 10
    fqe_batch_size: int = 128
    automl_tuning: AutoMLTuning = "off"
    write_plots: bool = False
    fail_fast: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if self.profile not in {"smoke", "core", "full", "paper"}:
            raise ValueError("profile must be 'smoke', 'core', 'full', or 'paper'.")
        if not self.seeds:
            raise ValueError("seeds must be nonempty.")
        if not self.families:
            raise ValueError("families must be nonempty.")
        for family in self.families:
            if family not in {"streamlift", "streamretain", "clinic_dtr", "epicare"}:
                raise ValueError(f"Unsupported family '{family}'.")
        for sample_size in self.sample_sizes:
            if int(sample_size) <= 0:
                raise ValueError("sample_sizes must be positive.")
        for gamma in self.gammas:
            if not (0.0 <= float(gamma) < 1.0):
                raise ValueError("gammas must be in [0, 1).")
        for horizon in self.observed_horizons:
            if int(horizon) <= 0:
                raise ValueError("observed_horizons must be positive.")
        for horizon in self.forecast_horizons:
            if int(horizon) <= 0:
                raise ValueError("forecast_horizons must be positive.")
        if int(self.trajectory_horizon) <= 0 or int(self.streamlift_long_horizon) <= 0:
            raise ValueError("trajectory horizons must be positive.")
        if int(self.streamlift_infinite_horizon_max_steps) <= 0:
            raise ValueError("streamlift_infinite_horizon_max_steps must be positive.")
        if int(self.mc_truth_rollouts) <= 0:
            raise ValueError("mc_truth_rollouts must be positive.")
        if not tuple(self.fqe_hidden_dims) or any(int(width) <= 0 for width in self.fqe_hidden_dims):
            raise ValueError("fqe_hidden_dims must contain positive widths.")
        if int(self.fqe_num_iterations) <= 0:
            raise ValueError("fqe_num_iterations must be positive.")
        if int(self.fqe_gradient_steps_per_iteration) <= 0:
            raise ValueError("fqe_gradient_steps_per_iteration must be positive.")
        if int(self.fqe_batch_size) <= 0:
            raise ValueError("fqe_batch_size must be positive.")
        if self.automl_tuning not in {"off", "fast", "balanced"}:
            raise ValueError("automl_tuning must be 'off', 'fast', or 'balanced'.")

    @classmethod
    def for_profile(
        cls,
        profile: Profile,
        *,
        output_root: str | Path = Path("outputs/causal_ope_benchmark"),
    ) -> "CausalOPEBenchmarkConfig":
        if profile == "smoke":
            return cls(
                profile="smoke",
                output_root=Path(output_root),
                seeds=(0,),
                families=("streamlift", "streamretain", "clinic_dtr"),
                sample_sizes=(120,),
                gammas=(0.90,),
                observed_horizons=(2, 3),
                target_policies=("moderate",),
                trajectory_horizon=18,
                streamlift_long_horizon=36,
                mc_truth_rollouts=48,
                fqe_hidden_dims=(32, 32),
                fqe_num_iterations=24,
                fqe_gradient_steps_per_iteration=10,
                fqe_batch_size=128,
            )
        if profile == "core":
            return cls(
                profile="core",
                output_root=Path(output_root),
                seeds=(0, 1),
                sample_sizes=(1000,),
                gammas=(0.95,),
                observed_horizons=(1, 3),
                target_policies=("conservative", "moderate", "safety_constrained"),
                trajectory_horizon=24,
                streamlift_long_horizon=36,
                mc_truth_rollouts=96,
                fqe_hidden_dims=(64, 64),
                fqe_num_iterations=48,
                fqe_gradient_steps_per_iteration=15,
                fqe_batch_size=256,
                automl_tuning="balanced",
            )
        if profile == "full":
            return cls(
                profile="full",
                output_root=Path(output_root),
                seeds=tuple(range(5)),
                sample_sizes=(500, 1500, 5000),
                gammas=(0.90, 0.95, 0.98),
                observed_horizons=(1, 2, 3),
                target_policies=("conservative", "moderate", "aggressive", "budget_constrained", "safety_constrained"),
                trajectory_horizon=36,
                streamlift_long_horizon=36,
                mc_truth_rollouts=320,
                fqe_hidden_dims=(128, 128),
                fqe_num_iterations=80,
                fqe_gradient_steps_per_iteration=20,
                fqe_batch_size=512,
                automl_tuning="balanced",
            )
        if profile == "paper":
            return cls(
                profile="paper",
                output_root=Path(output_root),
                seeds=tuple(range(5)),
                families=("streamretain", "clinic_dtr"),
                sample_sizes=(1_000, 5_000),
                gammas=(0.95,),
                observed_horizons=(3,),
                target_policies=("moderate", "safety_constrained"),
                trajectory_horizon=24,
                streamlift_long_horizon=36,
                mc_truth_rollouts=512,
                fqe_hidden_dims=(64, 64),
                fqe_num_iterations=64,
                fqe_gradient_steps_per_iteration=16,
                fqe_batch_size=256,
                automl_tuning="balanced",
            )
        raise ValueError("profile must be 'smoke', 'core', 'full', or 'paper'.")

    @classmethod
    def epicare_core_pilot(
        cls,
        *,
        output_root: str | Path = Path("outputs/causal_ope_benchmark"),
    ) -> "CausalOPEBenchmarkConfig":
        """Return the EpiCare tree-vs-neural core pilot configuration."""

        return cls(
            profile="core",
            output_root=Path(output_root),
            seeds=(0, 1, 2),
            families=("epicare",),
            sample_sizes=(500, 1500),
            gammas=(0.95,),
            observed_horizons=(3,),
            target_policies=("moderate", "safety_constrained"),
            estimators=(
                "boosted_fqe",
                "neural_fqe",
                "boosted_fqe_auto",
                "neural_fqe_auto",
                "discounted_occupancy_boosted",
                "discounted_occupancy_neural",
                "discounted_occupancy_boosted_auto",
                "discounted_occupancy_neural_auto",
            ),
            trajectory_horizon=24,
            mc_truth_rollouts=512,
            fqe_hidden_dims=(128, 128),
            fqe_num_iterations=64,
            fqe_gradient_steps_per_iteration=20,
            fqe_batch_size=256,
            automl_tuning="balanced",
        )

    def output_dir(self) -> Path:
        return self.output_root / self.profile


def scenarios_for_profile(profile: Profile, family: FamilyName) -> tuple[DomainScenario, ...]:
    """Return fixed scenario cells for a profile and family."""
    base = (
        DomainScenario(
            name="clean_randomized_good_overlap",
            overlap="good",
            confounding="randomized",
            delay_pattern="delayed_benefit" if family == "streamlift" else "immediate",
            missingness="none",
            censoring="administrative",
            noncompliance_rate=0.05 if family == "streamlift" else 0.0,
            target_policy_distance=0.30,
            surrogate_validity="valid",
        ),
    )
    if profile == "smoke":
        return base
    core = base + (
        DomainScenario(
            name="observed_moderate_overlap",
            overlap="moderate",
            confounding="observed",
            delay_pattern="short_harm_long_benefit" if family == "streamlift" else "delayed_benefit",
            missingness="none",
            censoring="administrative",
            noncompliance_rate=0.10 if family == "streamlift" else 0.0,
            target_policy_distance=0.60,
            surrogate_validity="valid",
        ),
    )
    if profile == "core":
        return core
    return core + (
        DomainScenario(
            name="observed_weak_overlap",
            overlap="weak",
            confounding="observed",
            delay_pattern="delayed_benefit",
            nonstationarity=False,
            missingness="none",
            censoring="administrative",
            noncompliance_rate=0.18 if family == "streamlift" else 0.0,
            subgroup_heterogeneity=1.0,
            target_policy_distance=0.75,
            surrogate_validity="valid",
        ),
    )


def sensitivity_scenarios_for_profile(profile: Profile, family: FamilyName) -> tuple[DomainScenario, ...]:
    """Return opt-in stress cells that violate or strain canonical assumptions."""
    if profile == "smoke":
        return ()
    return (
        DomainScenario(
            name="missing_censoring_nonstationary_stress",
            overlap="weak",
            confounding="observed",
            delay_pattern="short_benefit_long_harm" if family == "streamlift" else "delayed_benefit",
            nonstationarity=True,
            missingness="informative",
            censoring="competing_risk" if family == "clinic_dtr" else "informative",
            noncompliance_rate=0.18 if family == "streamlift" else 0.0,
            subgroup_heterogeneity=1.0,
            target_policy_distance=0.95,
            surrogate_validity="misleading",
            leaderboard_eligible=False,
        ),
        DomainScenario(
            name="latent_confounding_sensitivity",
            overlap="moderate",
            confounding="latent",
            delay_pattern="delayed_benefit",
            missingness="informative",
            censoring="informative",
            noncompliance_rate=0.12 if family == "streamlift" else 0.0,
            target_policy_distance=1.0,
            surrogate_validity="sign_reversal",
            leaderboard_eligible=False,
        ),
    )
