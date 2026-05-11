# Causal OPE Benchmark Release Checklist

Use this checklist before publishing `causal-ope-benchmark`.

1. Confirm the version in `pyproject.toml` and `causal_ope_benchmark/constants.py` match.
2. Update `CHANGELOG.md`, README, docs, and examples for any user-facing API or output-schema changes.
3. Run local quality gates:

```bash
python -m ruff check packages/causal-ope-benchmark
PYTHONPATH=packages/causal-ope-benchmark:packages/fqe:packages/occupancy-ratio \
  python -m pytest packages/causal-ope-benchmark/tests/test_causal_ope_benchmark.py
python packages/causal-ope-benchmark/scripts/packaging_smoke.py \
  --package-dir packages/causal-ope-benchmark
```

4. Inspect generated wheel/sdist metadata if needed:

```bash
python -m build packages/causal-ope-benchmark
```

5. Publish from a clean release checkout:

```bash
python -m twine upload packages/causal-ope-benchmark/dist/*
```

6. After upload, install from PyPI in a clean environment and run:

```bash
python -c "import causal_ope_benchmark as cob; print(cob.package_version())"
causal-ope-benchmark --list-families
causal-ope-benchmark --profile smoke --families streamretain --estimators direct_method
```
