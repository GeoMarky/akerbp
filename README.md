# SentinelAI — Interview Solution Scaffold

A small, readable mock of the SentinelAI case study on the open **SKAB** dataset.
Demonstrates deliberately weak **current controls** vs an improved design with:

- 1D-CNN detector
- **Temperature scaling** 
- Explicit **policy layer** (expected cost, dwell, hysteresis, abstain)

## Setup

```bash
cd solution
uv sync
uv run python -m sentinelai.data.download   # fetch SKAB into data/
uv run pytest
jupyter lab notebooks/
```

## Dev Container

Requires Docker. Open `solution/` in VS Code / Cursor and choose **Reopen in Container**.
Uses Python 3.14. Post-create runs `uv sync`, installs CPU PyTorch, downloads SKAB, and runs pytest.

```bash
uv run pytest
uv run jupyter lab notebooks/
```

## Layout

| Path | Purpose |
|------|---------|
| `src/sentinelai/data/windows.py` | Shared `make_window()` train/serve seam |
| `src/sentinelai/models/` | 1D-CNN detector + regression heads |
| `src/sentinelai/uncertainty/` | Calibration + conformal prediction |
| `src/sentinelai/policy/` | Load-bearing decision layer |
| `src/sentinelai/composer_policy/` | Load-bearing decision layer when you leave AI to write your code for you! |
| `src/sentinelai/baseline/` | Deliberately weak controls (before/after) |
| `notebooks/` | Walk through the main ideas |

## Notebooks

1. `01_explore_skab.ipynb` — data + gap injection
2. `02_calibration_policy.ipynb` — weak controls vs improved path (calibration, conformal, policy)
3. `04_quantile_heteroscedastic_newsensor.ipynb` — regression + new sensor: WIP - look at constraining uncertainty

## Mapping to the case

| Current control (weak) | Improved control |
|------------------------|------------------|
| Raw score as probability | Temperature scaling + ECE monitoring |
| Fixed 0.5 F1 threshold | Expected-cost decision rule |
| Forward-fill gaps | mask + dt channels + coverage gate |
| Auto-trip on Alarm | Dwell + hysteresis + abstain |
| No uncertainty monitoring | Conformal sets + audit records |

## References

**SKAB dataset**

```bibtex
@misc{skab,
  author = {Katser, Iurii D. and Kozitsin, Vyacheslav O.},
  title = {Skoltech Anomaly Benchmark (SKAB)},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/dsv/1693952}},
  DOI = {10.34740/KAGGLE/DSV/1693952}
}
```

**Conv_AE encoder** — The 1D-CNN detector encoder mirrors [SKAB's `Conv_AE`](https://github.com/waico/SKAB/blob/master/core/Conv_AE.py) (two strided `kernel_size=7` convolutions, `32 → 16` filters, `Dropout(0.2)`). The original implementation is TensorFlow/Keras; it was ported to PyTorch for this project (`src/sentinelai/models/cnn_detector.py`).

**OOD + conformal prediction** — Inspiration for the OOD test (using conformal prediction sets and abstain on ambiguous / out-of-distribution inputs):

```bibtex
@article{novello2024ood,
  author = {Novello, Paul and Dalmau, Joseba and Andeol, L{\'e}o},
  title = {Out-of-Distribution Detection Should Use Conformal Prediction (and Vice-versa?)},
  year = {2024},
  eprint = {2403.11532},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/pdf/2403.11532}
}
```


