# Bellman Aggregation Trees and Forests Draft

This directory contains a NeurIPS-style manuscript draft for target-weighted
Bellman Aggregation Trees and Forests (BATs/BAFs), non-iterative projected
Bellman estimators for offline fixed-policy evaluation under a declared
reference measure.

Build from this directory with:

```bash
TEXINPUTS="../../common/tex//:" latexmk -pdf main.tex
```

or from the repository root with:

```bash
make paper-bat
```

The current draft compiles to `main.pdf`. The experimental section is written
as a falsifiable protocol and mechanism-testing design; BAT/BAF-specific
numerical results are explicitly pending.
