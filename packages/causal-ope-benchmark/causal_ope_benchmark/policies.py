from __future__ import annotations

import numpy as np

from causal_ope_benchmark.types import Array, BenchmarkFamily, FixedPolicy, normalize_action_probs


STREAM_ACTIONS = (
    "no_op",
    "content_nudge",
    "push_email",
    "discount",
    "pause_offer",
    "downgrade_offer",
    "ad_load_reduction",
    "bundle_offer",
    "support_credit",
)
CLINIC_ACTIONS = (
    "monitor",
    "lifestyle",
    "intensify_cardio",
    "intensify_glycemic",
    "deintensify",
)
STREAMLIFT_ACTIONS = ("control", "treatment")
ACTION_NAMES_BY_FAMILY: dict[BenchmarkFamily, tuple[str, ...]] = {
    "streamlift": STREAMLIFT_ACTIONS,
    "streamretain": STREAM_ACTIONS,
    "clinic_dtr": CLINIC_ACTIONS,
}
POLICY_NAMES_BY_FAMILY: dict[BenchmarkFamily, tuple[str, ...]] = {
    "streamlift": ("moderate", "conservative", "aggressive"),
    "streamretain": ("moderate", "conservative", "aggressive", "budget_constrained", "safety_constrained"),
    "clinic_dtr": ("moderate", "conservative", "aggressive", "budget_constrained", "safety_constrained"),
    "epicare": ("moderate", "conservative", "aggressive", "budget_constrained", "safety_constrained"),
}


def get_fixed_policy(family: BenchmarkFamily, name: str) -> FixedPolicy:
    """Return a predeclared fixed target policy."""
    key = validate_policy_name(family, name)
    if family == "streamlift":
        return FixedPolicy(name=key, family=family, probability_fn=lambda states, availability, time: _streamlift_probs(key, states, availability))
    if family == "streamretain":
        return FixedPolicy(name=key, family=family, probability_fn=lambda states, availability, time: _streamretain_probs(key, states, availability, time))
    if family == "clinic_dtr":
        return FixedPolicy(name=key, family=family, probability_fn=lambda states, availability, time: _clinic_probs(key, states, availability))
    if family == "epicare":
        return FixedPolicy(name=key, family=family, probability_fn=lambda states, availability, time: _epicare_probs(key, states, availability, time))
    raise ValueError(f"Unsupported family '{family}'.")


def validate_policy_name(family: BenchmarkFamily, name: str) -> str:
    """Validate and normalize a fixed policy name for a benchmark family."""
    if family not in POLICY_NAMES_BY_FAMILY:
        raise ValueError(f"Unsupported family '{family}'.")
    key = str(name)
    allowed = POLICY_NAMES_BY_FAMILY[family]
    if key not in allowed:
        raise ValueError(f"Unknown target policy '{key}' for {family}. Allowed policies: {', '.join(allowed)}.")
    return key


def action_availability_for_family(family: BenchmarkFamily, states: Array) -> Array:
    """Return deterministic action availability for public benchmark states."""
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError("states must be a 2D array.")
    if family == "streamlift":
        return np.ones((s.shape[0], len(STREAMLIFT_ACTIONS)), dtype=np.float64)
    if family == "streamretain":
        return np.vstack([_stream_action_availability(row) for row in s]).astype(np.float64)
    if family == "clinic_dtr":
        return np.vstack([_clinic_action_availability(row) for row in s]).astype(np.float64)
    if family == "epicare":
        raise ValueError("EpiCare action availability depends on the external Gym environment; pass availability explicitly.")
    raise ValueError(f"Unsupported family '{family}'.")


def target_policy_probabilities(
    family: BenchmarkFamily,
    policy_name: str,
    states: Array,
    *,
    availability: Array | None = None,
    time: Array | None = None,
) -> Array:
    """Evaluate a named target policy over all discrete actions."""
    s = np.asarray(states, dtype=np.float64)
    avail = action_availability_for_family(family, s) if availability is None else np.asarray(availability, dtype=np.float64)
    t = np.zeros(s.shape[0], dtype=np.int64) if time is None else np.asarray(time)
    return get_fixed_policy(family, policy_name).probabilities(s, avail, t)


def _stream_action_availability(state: Array) -> Array:
    avail = np.ones(len(STREAM_ACTIONS), dtype=np.float64)
    if state[4] > 0.82:
        avail[2] = 0.0
    if state[2] < 0.25:
        avail[3] = 0.0
        avail[5] = 0.0
    if state[5] < 0.40:
        avail[4] = 0.0
        avail[8] = 0.0
    if state[3] < 0.30:
        avail[6] = 0.0
    return avail


def _clinic_action_availability(state: Array) -> Array:
    avail = np.ones(len(CLINIC_ACTIONS), dtype=np.float64)
    if state[6] > 0.72 or state[8] > 0.68:
        avail[2] = 0.0
        avail[3] = 0.0
    if state[3] > 0.75 or state[5] > 0.78:
        avail[4] = 0.0
    return avail


def _streamlift_probs(name: str, states: Array, availability: Array) -> Array:
    n = np.asarray(states).shape[0]
    if name == "conservative":
        p_treat = np.full(n, 0.35)
    elif name == "aggressive":
        p_treat = np.full(n, 0.75)
    else:
        p_treat = np.full(n, 0.55)
    probs = np.column_stack([1.0 - p_treat, p_treat])
    return normalize_action_probs(probs, availability)


def _streamretain_probs(name: str, states: Array, availability: Array, time: Array) -> Array:
    del time
    s = np.asarray(states, dtype=np.float64)
    engagement = s[:, 0]
    price_sens = s[:, 2]
    fatigue = s[:, 4]
    risk = s[:, 5]
    n = s.shape[0]
    probs = np.full((n, len(STREAM_ACTIONS)), 0.02, dtype=np.float64)
    probs[:, 0] = 0.50
    probs[:, 1] += 0.12 * (engagement < 0.45)
    probs[:, 2] += 0.08 * ((risk > 0.45) & (fatigue < 0.65))
    probs[:, 3] += 0.18 * (price_sens > 0.50)
    probs[:, 4] += 0.12 * (risk > 0.60)
    probs[:, 5] += 0.08 * (price_sens > 0.65)
    probs[:, 6] += 0.10 * ((risk > 0.45) & (price_sens < 0.45))
    probs[:, 7] += 0.06 * (engagement > 0.60)
    probs[:, 8] += 0.10 * (risk > 0.70)
    if name == "conservative":
        probs[:, 0] += 0.30
        probs[:, 3:] *= 0.55
    elif name == "aggressive":
        probs[:, 0] *= 0.35
        probs[:, 1:] *= 1.45
    elif name == "budget_constrained":
        probs[:, 3] *= 0.25
        probs[:, 8] *= 0.45
        probs[:, 1:3] *= 1.25
        probs[:, 0] += 0.15
    elif name == "safety_constrained":
        probs[:, 2] *= (fatigue < 0.55) + 0.20 * (fatigue >= 0.55)
        probs[:, 3] *= (risk > 0.50) + 0.15 * (risk <= 0.50)
        probs[:, 8] *= (risk > 0.60) + 0.10 * (risk <= 0.60)
        probs[:, 0] += 0.10
    return normalize_action_probs(probs, availability)


def _clinic_probs(name: str, states: Array, availability: Array) -> Array:
    s = np.asarray(states, dtype=np.float64)
    bp = s[:, 3]
    ldl = s[:, 4]
    hba1c = s[:, 5]
    kidney = s[:, 6]
    adherence = s[:, 7]
    toxicity = s[:, 8]
    n = s.shape[0]
    probs = np.full((n, len(CLINIC_ACTIONS)), 0.04, dtype=np.float64)
    probs[:, 0] = 0.36
    probs[:, 1] += 0.14 * (adherence < 0.55)
    probs[:, 2] += 0.24 * ((bp > 0.55) | (ldl > 0.55))
    probs[:, 3] += 0.24 * (hba1c > 0.55)
    probs[:, 4] += 0.18 * ((toxicity > 0.55) | (kidney > 0.70))
    if name == "conservative":
        probs[:, 0] += 0.25
        probs[:, 2:4] *= 0.70
    elif name == "aggressive":
        probs[:, 0] *= 0.45
        probs[:, 2:4] *= 1.50
    elif name == "budget_constrained":
        probs[:, 1] *= 1.40
        probs[:, 2:4] *= 0.80
        probs[:, 0] += 0.10
    elif name == "safety_constrained":
        high_risk = (kidney > 0.68) | (toxicity > 0.50)
        probs[:, 2:4] *= (~high_risk[:, None]) + 0.20 * high_risk[:, None]
        probs[:, 4] += 0.20 * high_risk
    return normalize_action_probs(probs, availability)


def _epicare_probs(name: str, states: Array, availability: Array, time: Array) -> Array:
    """Generic named policies for the external EpiCare action space.

    EpiCare owns its clinical simulator and action semantics. These policies
    only provide reproducible, non-oracle probability surfaces over whatever
    discrete action count the installed environment exposes.
    """

    del time
    s = np.asarray(states, dtype=np.float64)
    avail = np.asarray(availability, dtype=np.float64)
    if avail.ndim != 2:
        raise ValueError("availability must be a 2D action mask for EpiCare policies.")
    n, n_actions = avail.shape
    if n_actions < 1:
        raise ValueError("EpiCare policies require at least one action.")
    summary = np.zeros(n, dtype=np.float64)
    if s.ndim == 2 and s.shape[1] > 0:
        usable = np.nan_to_num(s[:, : min(6, s.shape[1])], nan=0.0, posinf=1.0, neginf=0.0)
        summary = np.clip(np.mean(usable, axis=1), 0.0, 1.0)
    action_rank = np.linspace(0.0, 1.0, n_actions, dtype=np.float64)
    probs = np.full((n, n_actions), 0.05, dtype=np.float64)
    probs[:, 0] = 0.55
    if n_actions > 1:
        probs[:, 1:] += 0.20 * summary[:, None] * action_rank[1:][None, :]
    if name == "conservative":
        probs[:, 0] += 0.35
        if n_actions > 1:
            probs[:, 1:] *= 0.55
    elif name == "aggressive":
        probs[:, 0] *= 0.35
        if n_actions > 1:
            probs[:, 1:] *= 1.65
    elif name == "budget_constrained":
        probs[:, 0] += 0.20
        if n_actions > 2:
            probs[:, 2:] *= 0.75
    elif name == "safety_constrained":
        probs[:, 0] += 0.15
        if n_actions > 1:
            high_risk = summary > 0.70
            probs[high_risk, 1:] *= 0.60
    return normalize_action_probs(probs, avail)
