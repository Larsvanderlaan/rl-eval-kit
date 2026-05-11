#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from FQE_calibration_neurips.scripts.run_experiment import run_config  # noqa: E402
from FQE_calibration_neurips.src.aggregation import aggregate_results  # noqa: E402
from FQE_calibration_neurips.src.utils import ensure_dir, load_config, write_json  # noqa: E402
from FQE_calibration_neurips.src.validation import (  # noqa: E402
    GateConfig,
    apply_gate_to_rows,
    evaluate_well_specified_gate,
)


def _resolve_path(path: str, suite_config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    local = suite_config_path.parent / candidate
    if local.exists():
        return local
    return ROOT / path


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_suite_config(entry: dict[str, Any], suite_config_path: Path, mode: str, replications: int | None) -> dict[str, Any]:
    cfg = load_config(_resolve_path(str(entry["config"]), suite_config_path))
    cfg = _deep_merge(cfg, entry.get("overrides", {}))
    cfg = _deep_merge(cfg, entry.get("mode_overrides", {}).get(mode, {}))
    if replications is not None:
        cfg["replications"] = int(replications)
    return cfg


def _validation_replications(validation_entry: dict[str, Any], mode: str, suite_replications: int | None) -> int | None:
    by_mode = validation_entry.get("replications_by_mode", {})
    if isinstance(by_mode, dict) and mode in by_mode:
        return int(by_mode[mode])
    return suite_replications


def _write_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "raw_results.csv", index=False)


def run_suite(
    suite_config_path: str | Path,
    *,
    mode: str,
    replications: int | None = None,
    continue_on_failure: bool = False,
    skip_validation_gate: bool = False,
    results_dir: str | Path | None = None,
) -> Path:
    suite_config_path = Path(suite_config_path)
    suite_cfg = load_config(suite_config_path)
    if results_dir is not None:
        mode_dir = ensure_dir(Path(results_dir))
    else:
        output_subdir = suite_cfg.get("output_subdir", mode)
        if isinstance(output_subdir, dict):
            output_subdir = output_subdir.get(mode, mode)
        mode_dir = ensure_dir(ROOT / "results" / str(output_subdir))
    raw_dir = mode_dir / "raw"
    validation_dir = mode_dir / "validation"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    if validation_dir.exists():
        shutil.rmtree(validation_dir)
    ensure_dir(raw_dir)
    ensure_dir(validation_dir)

    validation_entry = suite_cfg.get("validation", {})
    gate_frame = pd.DataFrame()
    gate_passed = True
    if not skip_validation_gate:
        val_name = str(validation_entry.get("name", "well_specified_debug"))
        val_reps = _validation_replications(validation_entry, mode, replications)
        val_cfg = _load_suite_config(validation_entry, suite_config_path, mode, val_reps)
        val_output = mode_dir / "raw" / val_name
        val_rows = run_config(val_cfg, val_output, debug=False, run_mode=mode, suite_name=val_name)
        gate_cfg = GateConfig(**validation_entry.get("gate", {}))
        gate_passed, gate_frame = evaluate_well_specified_gate(val_rows, mode_dir / "validation", gate_cfg)
        write_json(val_output / "suite_metadata.json", {"suite_name": val_name, "mode": mode, "validation_gate": True})
        if mode in {"paper", "final"} and not gate_passed:
            raise RuntimeError(
                f"Validation gate failed for {mode} mode. See {mode_dir / 'validation'}. "
                "Use --skip_validation_gate only for explicit diagnostics."
            )

    for entry in suite_cfg.get("suites", []):
        name = str(entry["name"])
        if mode in set(entry.get("skip_modes", [])):
            audit_path = mode_dir / "validation" / "audit_notes.md"
            with audit_path.open("a") as handle:
                handle.write(f"\n\n## Suite Skipped: `{name}`\n\nSkipped in `{mode}` mode by suite config.\n")
            continue
        output_dir = mode_dir / "raw" / name
        try:
            cfg = _load_suite_config(entry, suite_config_path, mode, replications)
            rows = run_config(cfg, output_dir, debug=False, run_mode=mode, suite_name=name)
            if not gate_frame.empty:
                rows = apply_gate_to_rows(rows, gate_frame)
                _write_rows(rows, output_dir)
            write_json(output_dir / "suite_metadata.json", {"suite_name": name, "mode": mode, "validation_gate_passed": gate_passed})
        except Exception as exc:
            audit_path = mode_dir / "validation" / "audit_notes.md"
            with audit_path.open("a") as handle:
                handle.write(f"\n\n## Suite Failure: `{name}`\n\n`{type(exc).__name__}: {exc}`\n")
            if not continue_on_failure:
                raise
    aggregate_results(mode_dir, write_tables=True)
    return mode_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a gated suite of FQE calibration experiments.")
    parser.add_argument("--suite_config", type=str, default=str(ROOT / "configs/paper_suite.yaml"))
    parser.add_argument(
        "--mode",
        choices=["debug", "pilot", "confirm", "final", "paper"],
        default="debug",
        help="Run mode. pilot/confirm/final are staged rescue modes; paper is the legacy final mode.",
    )
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--continue_on_failure", action="store_true")
    parser.add_argument("--skip_validation_gate", action="store_true")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to FQE_calibration_neurips/results/<mode> or suite output_subdir.",
    )
    args = parser.parse_args()
    out = run_suite(
        args.suite_config,
        mode=args.mode,
        replications=args.replications,
        continue_on_failure=args.continue_on_failure,
        skip_validation_gate=args.skip_validation_gate,
        results_dir=args.results_dir,
    )
    print(f"Wrote suite outputs to {out}")


if __name__ == "__main__":
    main()
