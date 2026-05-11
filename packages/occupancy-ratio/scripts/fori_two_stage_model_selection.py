#!/usr/bin/env python3
"""Run FORI two-stage ABE model selection from a JSON/YAML config.

The CLI intentionally keeps data loading minimal: provide an ``.npz`` file with
array keys matching the ``FORITwoStageCV.fit`` arguments, and a config file with
``gamma`` plus optional ``model_selection`` config overrides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

from occupancy_ratio.fori_model_selection import (
    FORICandidateSpec,
    FORITwoStageCV,
    FORITwoStageCVConfig,
    load_fori_two_stage_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FORI two-stage ABE model selection.")
    parser.add_argument("--config", required=True, help="JSON/YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = dict(load_fori_two_stage_config(args.config))
    dataset_path = payload.get("dataset_path")
    if dataset_path is None:
        raise ValueError("Config must include dataset_path pointing to an .npz file.")
    data = np.load(Path(dataset_path), allow_pickle=True)
    model_cfg = dict(payload.get("model_selection", {}))
    config = FORITwoStageCVConfig(**model_cfg)
    candidates = _load_cached_candidates(payload.get("candidates", []), data)
    fit_kwargs: dict[str, Any] = {
        "states": _required(data, "states"),
        "actions": _required(data, "actions"),
        "next_states": _required(data, "next_states"),
        "target_actions": _optional(data, "target_actions"),
        "target_next_actions": _optional(data, "target_next_actions"),
        "gamma": float(payload.get("gamma", data["gamma"] if "gamma" in data else 0.99)),
        "episode_ids": _required(data, "episode_ids"),
        "rewards": _optional(data, "rewards"),
        "initial_states": _optional(data, "initial_states"),
        "initial_actions": _optional(data, "initial_actions"),
        "initial_weights": _optional(data, "initial_weights"),
        "initial_episode_ids": _optional(data, "initial_episode_ids"),
        "terminated": _optional(data, "terminated"),
        "truncated": _optional(data, "truncated"),
        "done": _optional(data, "done"),
        "candidates": candidates,
    }
    result = FORITwoStageCV(config).fit(**fit_kwargs)
    output_path = Path(payload.get("output_path", "fori_two_stage_model_selection_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "selected_candidate_id": result.selected_candidate_id,
        "selection_rule": result.selection_rule,
        "selected_min_score_candidate_id": result.selected_min_score_candidate_id,
        "selected_one_se_candidate_id": result.selected_one_se_candidate_id,
        "warnings": result.warnings,
        "rows": result.candidate_rows(),
        "first_stage_diagnostics": dict(result.first_stage.diagnostics),
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"selected_candidate_id={result.selected_candidate_id} ({result.selection_rule})")
    print(f"selected_min_score={result.selected_min_score_candidate_id}")
    print(f"selected_one_se={result.selected_one_se_candidate_id}")
    return 0


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Dataset .npz must contain {key!r}.")
    return data[key]


def _optional(data: Mapping[str, Any], key: str) -> Any:
    return data[key] if key in data else None


def _load_cached_candidates(rows: Any, data: Mapping[str, Any]) -> list[FORICandidateSpec]:
    specs: list[FORICandidateSpec] = []
    if not rows:
        return specs
    if not isinstance(rows, list):
        raise ValueError("candidates must be a list of cached candidate specs.")
    for pos, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("Each candidate spec must be an object.")
        candidate_id = str(row.get("candidate_id", f"candidate_{pos:03d}"))
        prediction_key = row.get("prediction_key")
        cached_predictions = None
        if prediction_key is not None:
            if prediction_key not in data:
                raise ValueError(f"prediction_key {prediction_key!r} is missing from dataset .npz.")
            cached_predictions = data[prediction_key]
        split_prediction_keys = row.get("split_prediction_keys")
        if isinstance(split_prediction_keys, Mapping):
            cached_predictions = {
                str(split): data[str(key)]
                for split, key in split_prediction_keys.items()
            }
        specs.append(
            FORICandidateSpec(
                candidate_id=candidate_id,
                family=str(row.get("family", "external")),
                cached_predictions=cached_predictions,
                hyperparams=dict(row.get("hyperparams", {})),
                metadata=dict(row.get("metadata", {})),
                complexity_order_key=row.get("complexity_order_key"),
                iteration=row.get("iteration"),
                projection_type=str(row.get("projection_type", "")),
                damping_alpha=row.get("damping_alpha"),
            )
        )
    return specs


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
