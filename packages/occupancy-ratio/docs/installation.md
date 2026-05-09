# Installation

Install from the package directory:

```bash
python -m pip install -e "packages/occupancy-ratio"
```

Common extras:

```bash
python -m pip install -e "packages/occupancy-ratio[neural]"
python -m pip install -e "packages/occupancy-ratio[benchmark]"
python -m pip install -e "packages/occupancy-ratio[docs]"
python -m pip install -e "packages/occupancy-ratio[neural,benchmark,docs]"
```

## Extras

| Extra | Adds | Use when |
| --- | --- | --- |
| `neural` | PyTorch | You need `fit_discounted_occupancy_ratio_neural(...)`. |
| `benchmark` | Gymnasium, MuJoCo, plotting | You run `occupancy-ratio-benchmark`. |
| `tabular-benchmark` | OpenML, OBP, Minari | You run optional logged/tabular benchmark settings. |
| `google` / `google-dualdice` | TensorFlow, TensorFlow Addons | You compare against official Google DualDICE. |
| `docs` | MkDocs Material and mkdocstrings | You build this documentation site. |
| `dev` | Test and development dependencies | You work on the package. |

## Build The Docs

```bash
python -m mkdocs build --strict -f packages/occupancy-ratio/mkdocs.yml
```

For local preview:

```bash
python -m mkdocs serve -f packages/occupancy-ratio/mkdocs.yml
```

The docs import the package through the package root, so editable installation
is recommended before building API pages.
