"""Build, install, and smoke-test the causal-ope-benchmark package.

This script is intentionally dependency-light. It creates a temporary virtual
environment with access to the current interpreter's site packages so CI/local
smoke runs can reuse already-installed NumPy while still installing the built
wheel as a real package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and smoke-test causal-ope-benchmark.")
    parser.add_argument("--package-dir", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="causal_ope_pkg_smoke_") as tmp:
        tmp_dir = Path(tmp)
        dist_dir = tmp_dir / "dist"
        _run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist_dir), str(package_dir)])
        wheels = sorted(dist_dir.glob("causal_ope_benchmark-*.whl"))
        if not wheels:
            raise SystemExit("No wheel was built.")
        env_dir = tmp_dir / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        _run([str(python), "-m", "pip", "install", "--no-deps", str(wheels[-1])])
        _run([str(python), "-c", "import causal_ope_benchmark as cob; print(cob.package_version())"])
        _run([str(python), "-m", "causal_ope_benchmark.run", "--list-families"])
        output_root = tmp_dir / "outputs"
        _run(
            [
                str(python),
                "-m",
                "causal_ope_benchmark.run",
                "--profile",
                "smoke",
                "--families",
                "streamretain",
                "--estimators",
                "direct_method",
                "--sample-sizes",
                "8",
                "--mc-truth-rollouts",
                "2",
                "--output-root",
                str(output_root),
            ]
        )


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
