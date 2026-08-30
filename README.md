# How Much Does the Wrong SAE Cost?

Anonymized code release for a NeurIPS 2026 Interp4Discovery submission ("How Much Does
the Wrong SAE Cost? Calibrating Quality Metrics Against Board-Game Ground Truth"),
provided for double-blind review. Using 80 SAEs trained on OthelloGPT, the paper shows
that the standard proxy metrics (L0, L1, FVU, dead-feature rate) separate collapsed
dictionaries from healthy ones without error, but that ranking within the healthy set is
a smaller question than it looks: the shortfall from the best available dictionary is
under one point of ground-truth quality, comparable to what drawing three dictionaries
at random already achieves. The paper argues that a collapse screen paired with a
shortfall metric answers the selection question that rank agreement does not.

## What's in this release

This is a curated, code-and-results release, not the full project repo. It contains:

- `analysis/reproduce_paper.py` — the single entry point. Prints every number reported
  in the paper (main text and appendices), computed from the results below.
- `analysis/figure_screening.py`, `analysis/figure_selection.py` — regenerate the two
  figures (a print statement can't reproduce a plot).
- `analysis/compare_split_conventions.py` — standalone deeper diagnostic behind the
  probe train/test split comparison in Appendix D (rank movement, per-architecture
  breakdown, reseed floors under both split conventions).
- `src/boardsae/` — the package these scripts import: tidy-results I/O, axis-discovery
  (parallel analysis, factor analysis), and metric-validity code (partial correlation,
  reseed noise floor). This is the full package as used internally; only the
  `discovery`, `validity`, and `io` subpackages are on the path the scripts above
  actually exercise.
- `results/records.parquet`, `results/records_grouped_split.parquet` — the full,
  already-computed population run (SAE x ground-truth quantity x proxy metric x seed).
  Every number in the paper is read off these two files.
- `tests/unit/` — the subset of the project's unit tests that covers `discovery`,
  `validity`, `io`, and SAE-population sampling: pure functions over synthetic or
  already-computed data, no torch and no downloaded model or SAE checkpoints.

**Deliberately excluded**: the model-loading, SAE-loading, and population-run pipeline
that produced `results/*.parquet` in the first place. That path needs a GPU-scale
transformer_lens/torch environment and the released OthelloGPT SAE sweep, and takes on
the order of an hour to run; reproducing the reported *numbers* does not require
re-running it, since the results it produced are included directly. Also excluded:
exploratory/presentation figure scripts not tied to a reported number, the acceptance
test suite (needs the real model and the downloaded SAE sweep), and the unit tests for
the excluded model/SAE-loading modules (they import `torch`, which this release does
not depend on).

## Setup

Verified against Python 3.11 (the version pinned in the full project) and 3.12 (a clean
venv with only the packages below). No GPU, `torch`, or `transformer_lens` needed for
anything in this release, unlike the excluded population pipeline.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `factor_analyzer`'s current PyPI release calls a scikit-learn function removed in
scikit-learn 1.8. `requirements.txt` pins scikit-learn below that version; installing an
unpinned or newer scikit-learn instead raises a `TypeError` from inside
`factor_analyzer.FactorAnalyzer`, not from this code.

## Run

All four scripts insert `src/` onto `sys.path` themselves and resolve `results/*.parquet`
relative to the current directory, so run them from the repository root with a plain
`python` call, no `PYTHONPATH` or install of this package needed:

```bash
python analysis/reproduce_paper.py
```

This reads `results/records_grouped_split.parquet` by default and prints, in order:
the Table 1 screening ranges per proxy metric, shortfall-at-k with random-draw
baselines, the screening gap arithmetic, the Appendix D ground-truth two-group ranges,
the positive-manifold summary over the composite correlation matrix, the sensitivity
table matching Table 1's caption, and a headline comparison against the split
convention in `results/records.parquet`. No arguments are needed; every number printed
should match the corresponding value in the paper.

To regenerate the two figures:

```bash
python analysis/figure_screening.py
python analysis/figure_selection.py
```

Both write to `figures/` (created if missing).

To run the standalone split-convention diagnostic:

```bash
python analysis/compare_split_conventions.py
```

## Tests

```bash
python -m pytest tests/unit
```

Runs offline, on synthetic or already-computed data only.
