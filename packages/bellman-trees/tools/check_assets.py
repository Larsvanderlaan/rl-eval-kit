#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "papers" / "neurips_bellman"
PAPERS = [
    ROOT / "papers" / "fqe",
    ROOT / "papers" / "calibration",
    ROOT / "papers" / "soft_fqi_stationary_weighting",
    ROOT / "papers" / "bellman_aggregation_trees",
]

COMMAND_RE = re.compile(
    r"\\(?P<cmd>includegraphics|input|include|bibliography|addbibresource)"
    r"(?:\s*\[[^\]]*\])*\s*\{(?P<arg>[^}]*)\}"
)


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        out: list[str] = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            out.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def candidate_paths(base: Path, command: str, arg: str) -> list[Path]:
    raw = arg.strip()
    if not raw:
        return []
    if command in {"bibliography", "addbibresource"}:
        out: list[Path] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            path = base / part
            out.append(path if path.suffix else path.with_suffix(".bib"))
        return out
    path = base / raw
    if command in {"input", "include"} and not path.suffix:
        return [path.with_suffix(".tex")]
    if command == "includegraphics" and not path.suffix:
        return [path.with_suffix(ext) for ext in (".pdf", ".png", ".jpg", ".jpeg")]
    return [path]


def main() -> int:
    missing: list[str] = []
    referenced_figures: set[Path] = set()
    referenced_figure_stems: set[Path] = set()

    for paper in PAPERS:
        tex = paper / "main.tex"
        if not tex.exists():
            missing.append(str(tex.relative_to(ROOT)))
            continue
        text = strip_comments(tex.read_text(encoding="utf-8"))
        for match in COMMAND_RE.finditer(text):
            command = match.group("cmd")
            for path in candidate_paths(paper, command, match.group("arg")):
                if command == "includegraphics":
                    if not path.exists():
                        alternatives = candidate_paths(paper, command, match.group("arg"))
                        if not any(alt.exists() for alt in alternatives):
                            missing.append(str(path.relative_to(ROOT)))
                    else:
                        referenced_figures.add(path.resolve())
                        referenced_figure_stems.add(path.with_suffix("").resolve())
                elif not path.exists():
                    missing.append(str(path.relative_to(ROOT)))

    unreferenced: list[str] = []
    for paper in PAPERS:
        fig_dir = paper / "figures"
        if not fig_dir.exists():
            continue
        for path in fig_dir.iterdir():
            if (
                path.is_file()
                and path.resolve() not in referenced_figures
                and path.with_suffix("").resolve() not in referenced_figure_stems
            ):
                unreferenced.append(str(path.relative_to(ROOT)))

    if missing or unreferenced:
        if missing:
            print("Missing referenced files:")
            for path in sorted(set(missing)):
                print(f"  - {path}")
        if unreferenced:
            print("Unreferenced paper figures:")
            for path in sorted(unreferenced):
                print(f"  - {path}")
        return 1

    print("All canonical paper references resolve; no unreferenced paper figures found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
