# Protocol Before Architecture: BreaKHis code, predictions and analyses

Code and results for the paper

> Rizvi QM, Talat S, Rizvi IM, Ansari MA, Ahmad K, Jamal QMS. **Protocol Before Architecture: Data Leakage,
> Variance and Trustworthiness in Breast Cancer Histopathology Classification.** Manuscript under review, 2026.

Five architectures were trained with patient-wise five-fold cross-validation at all four BreaKHis magnifications
(20 runs): a bidirectional cross-attention CNN–Transformer hybrid (BCAF), its concatenation ablation,
ConvNeXt-Tiny, Swin-Tiny and ResNet-50. Four further runs repeat the protocol with image-wise ("leaky")
partitioning. The repository holds the following:

* the exact training and evaluation code that produced every result (`notebooks/`, `src/`), including the
  **executable patient-disjointness assertions**;
* the **out-of-fold predictions of all 20 patient-wise runs**: one row per image with the true label, prediction,
  class probabilities and logits;
* the scripts that recompute every statistic, table value and figure in the paper from those predictions, without
  retraining (`analysis/`).

The BreaKHis images are **not** included. Download them from the
[original source (UFPR)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
or from [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis).

## Reproduce every number in the paper (CPU, about 10 minutes, no images needed)

```bash
pip install numpy pandas scipy scikit-learn matplotlib
python analysis/sweep_analyses.py     # bootstrap, power/MDD, risk-coverage, reallocation simulation (~7 min)
python analysis/unit_of_analysis.py   # naive vs cluster-aware vs patient-level tests (Table 4)
python analysis/quantities.py         # prints every quantity quoted in the manuscript
python analysis/sweep_figures.py      # regenerates the result figures into figures/
```

The outputs are already committed in `results/derived/` and `figures/`, and
[`results/derived/quantities_report.txt`](results/derived/quantities_report.txt) is the full printed report.

| Paper item | Source |
|---|---|
| Table 1 (this work, both protocols) | `results/sweep/performance_by_magnification.csv`, `leakage_by_magnification.csv` |
| Table 2: leakage at every magnification | `results/sweep/leakage_by_magnification.csv`, `leakage_m*.csv` |
| Table 3: balanced accuracy, 5 models × 4 magnifications | `results/sweep/fold_metrics_*_m*.csv` (report section 1) |
| Table 4: forty pairwise comparisons, resolving power | `results/derived/unit_of_analysis.csv`, `power_all.csv`, `patient_bootstrap_diff.csv`, `results/sweep/stats_family_by_magnification.csv`, `stats_equivalence_power.csv` |
| Table 5 and Table S4: calibration and selective prediction | `results/sweep/calibration_per_fold.csv`, `results/derived/risk_coverage.csv`, `selective_prediction.csv` |
| Table 6: explanation faithfulness | `results/sweep/xai_faithfulness_allfolds.csv` |
| Variance origin and reallocation simulation | `results/derived/patient_error_concentration.csv`, `fold_spread_simulation.csv`, `fold_composition_corr.csv` |
| Tables S1–S3 | `results/sweep/dataset_verification.csv`, `fold_composition_m*.csv`, `results/derived/patient_bootstrap_ci.csv` |

The cluster-aware permutation p-values are Monte Carlo estimates (20,000 sign flips). The counts of significant
comparisons do not depend on the seed. The median ratio of cluster-aware to naive p-values comes out at 11.8 with
the seed used here and at 11.9 in the paper.

## Retrain (GPU)

`notebooks/BCAF_BreaKHis_FULL_SWEEP.ipynb` is the notebook that ran the sweep: 20 patient-wise runs, 4 image-wise
runs, fold-honest calibration and explanation faithfulness. It runs on Kaggle or Google Colab. The notebook writes
its own modules to `src/` at the start. The copies in `src/` are the same code, extracted for reading and reuse:

| Module | Contents |
|---|---|
| `src/data.py` | Indexing from the BreaKHis filename grammar, the patient grouping key, stratified group k-fold, and the leakage assertions `assert_no_patient_leakage` and `assert_disjoint`. Both raise an error rather than warn. |
| `src/models.py` | BCAF, the concatenation ablation and the baselines |
| `src/engine.py` | One training and evaluation path for every architecture |
| `src/calibration.py` | ECE, Brier score and leave-one-fold-out temperature scaling |
| `src/xai.py` | Grad-CAM, Grad-CAM++, fusion attention, deletion/insertion, AOPC, ADCC and the random-saliency control |
| `src/stats.py` | Corrected resampled t-test, Wilcoxon test, TOST, Holm correction and bootstrap |

Training dependencies: `pip install -r requirements.txt` (PyTorch, torchvision and timm). The 20 patient-wise runs
took 10.4 GPU-hours on one NVIDIA L4. The training settings are in `results/sweep/config.json`.

To fetch and check the dataset locally, run `bash scripts/get_breakhis.sh ./data`, then
`python scripts/verify_dataset.py ./data`.

## Notes

* Patient key: two malignant slide folders (`SOB_M_DC_14-13412` and `SOB_M_LC_14-13412`) share the identifier
  14-13412 and are treated as one grouping unit. This gives 81 units at every magnification.
* Model selection used balanced accuracy on the evaluation fold, for all architectures alike. Absolute values are
  therefore slightly optimistic. Comparisons between models are not affected.

## Licence

Code: MIT (see `LICENSE`). The prediction files are derived from BreaKHis and are released for research use.
Cite the paper above and the original dataset (Spanhol et al., *IEEE Trans Biomed Eng* 2016;63(7):1455–1462).
