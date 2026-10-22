# How Much Does the Wrong SAE Cost?

Code and results for "How Much Does the Wrong SAE Cost? Calibrating Quality Metrics
Against Board-Game Ground Truth", Sofia B. Sartori, NeurIPS 2026 Workshop on
Interpretability for Discovery (Interp4Discovery).

Using 80 SAEs trained on OthelloGPT, the paper shows that the standard proxy metrics (L0,
L1, FVU, dead-feature rate) separate collapsed dictionaries from healthy ones without
error, but that ranking within the healthy set is a smaller question than it looks. The
shortfall from the best available dictionary is about one point of ground-truth quality,
less than what drawing three dictionaries at random achieves. The paper argues that a
collapse screen paired with a shortfall metric answers the selection question that rank
agreement does not.

## What's in this release

This is a curated code-and-results release, not the full project repository. It contains
the following.

- `analysis/reproduce_paper.py`, the single entry point. It prints every number reported
  in the paper, main text and appendices, computed from the results below.
- `analysis/figure_screening.py` and `analysis/figure_width.py`, which regenerate
  Figure 1 (screening and shortfall) and Figure 2 (ground truth against dictionary width).
- `analysis/figure_selection.py`, an earlier view of the shortfall curves that is not in
  the paper, kept because it is cheap to regenerate.
- `analysis/compare_split_conventions.py`, the standalone diagnostic behind the probe
  train/test split comparison in Appendix C (rank movement, per-architecture breakdown,
  reseed floors under both split conventions).
- `src/boardsae/`, the package these scripts import: tidy-results I/O, axis discovery
  (parallel analysis, factor analysis), and metric-validity code (partial correlation,
  reseed noise floor). This is the full package as used internally; only the
  `discovery`, `validity`, and `io` subpackages are on the path the scripts above
  exercise.
- `results/records.parquet` and `results/records_grouped_split.parquet`, the full,
  already-computed population run (SAE x ground-truth quantity x proxy metric x seed).
  Every number in the paper is read off these two files; the grouped-split file is the
  one the paper reports.
- `tests/unit/`, the subset of the project's unit tests that covers `discovery`,
  `validity`, `io`, and SAE-population sampling. These are pure functions over synthetic
  or already-computed data, with no torch and no downloaded model or SAE checkpoints.

**Deliberately excluded:** the model-loading, SAE-loading, and population-run pipeline
that produced `results/*.parquet`. That path needs a GPU-scale transformer_lens/torch
environment and the released OthelloGPT SAE sweep, and takes on the order of an hour to
run. Reproducing the reported numbers does not require re-running it, since its results
are included directly. Also excluded are exploratory and presentation figure scripts not
tied to a reported number, the acceptance test suite (it needs the real model and the
downloaded SAE sweep), and the unit tests for the excluded model and SAE-loading modules
(they import `torch`, which this release does not depend on).

## Setup

Verified against Python 3.11 (the version pinned in the full project) and 3.12 (a clean
venv with only the packages below). No GPU, `torch`, or `transformer_lens` is needed for
anything in this release.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`factor_analyzer`'s current PyPI release calls a scikit-learn function removed in
scikit-learn 1.8. `requirements.txt` pins scikit-learn below that version; installing an
unpinned or newer scikit-learn raises a `TypeError` from inside
`factor_analyzer.FactorAnalyzer`, not from this code.

## Run

All scripts insert `src/` onto `sys.path` themselves and resolve `results/*.parquet`
relative to the current directory, so run them from the repository root with a plain
`python` call, with no `PYTHONPATH` or install of this package.

```bash
python analysis/reproduce_paper.py
```

This reads `results/records_grouped_split.parquet` by default and prints one section per
reported claim, each tagged with the paper section or table it belongs to. That covers
the factor-structure results, the Table 1 screening ranges and shortfalls with
random-draw baselines, the validity and sensitivity tables (Tables 4 and 5), the reseed
floor (Table 3), the population table (Table 2), the appendix controls, and the
camera-ready additions: shortfall within each reseed, shortfall in next-move KL, the cost
of choosing within a metric's top three without ground truth, and ground truth against
dictionary width. It ends with a headline comparison against the row-level split in
`results/records.parquet`. No arguments are needed.

To regenerate the figures:

```bash
python analysis/figure_screening.py
python analysis/figure_width.py
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

## Reused artifacts

The SAEs are the released OthelloGPT sweep of Karvonen et al. (2024), "Measuring
Progress in Dictionary Learning for Language Model Interpretability with Board Game
Models". The model is OthelloGPT (Li et al., 2023). Neither is redistributed here; only
the measurements computed from them are.

## Citation

```bibtex
@inproceedings{sartori2026wrongsae,
  title     = {How Much Does the Wrong {SAE} Cost? Calibrating Quality Metrics Against
               Board-Game Ground Truth},
  author    = {Sartori, Sofia B.},
  booktitle = {NeurIPS 2026 Workshop on Interpretability for Discovery},
  year      = {2026}
}
```

## License

[TO BE CHOSEN]
