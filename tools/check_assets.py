#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NEURIPS_BELLMAN_ROOT = REPO_ROOT / "papers" / "neurips_bellman"
PAPERS = [
    NEURIPS_BELLMAN_ROOT / "papers" / "fqe" / "main.tex",
    NEURIPS_BELLMAN_ROOT / "papers" / "calibration" / "main.tex",
    NEURIPS_BELLMAN_ROOT / "papers" / "soft_fqi_stationary_weighting" / "main.tex",
    NEURIPS_BELLMAN_ROOT / "papers" / "bellman_aggregation_trees" / "main.tex",
    REPO_ROOT / "papers" / "irl" / "journal_debiased_irl" / "main_jasa.tex",
    REPO_ROOT / "papers" / "irl" / "journal_debiased_irl" / "main_jasa_appendix.tex",
    REPO_ROOT / "papers" / "irl" / "conference_genpqr" / "main_neurips.tex",
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


def display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    missing: list[str] = []
    referenced_figures: set[Path] = set()
    referenced_figure_stems: set[Path] = set()

    for tex in PAPERS:
        paper = tex.parent
        if not tex.exists():
            missing.append(display(tex))
            continue
        text = strip_comments(tex.read_text(encoding="utf-8"))
        for match in COMMAND_RE.finditer(text):
            command = match.group("cmd")
            for path in candidate_paths(paper, command, match.group("arg")):
                if command == "includegraphics":
                    alternatives = candidate_paths(paper, command, match.group("arg"))
                    found = next((alt for alt in alternatives if alt.exists()), None)
                    if found is None:
                        missing.append(display(path))
                    else:
                        referenced_figures.add(found.resolve())
                        referenced_figure_stems.add(found.with_suffix("").resolve())
                elif not path.exists():
                    missing.append(display(path))

    unreferenced: list[str] = []
    for tex in PAPERS:
        paper = tex.parent
        fig_dir = paper / "figures"
        if not fig_dir.exists():
            continue
        for path in fig_dir.iterdir():
            if (
                path.is_file()
                and path.resolve() not in referenced_figures
                and path.with_suffix("").resolve() not in referenced_figure_stems
            ):
                unreferenced.append(display(path))

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
