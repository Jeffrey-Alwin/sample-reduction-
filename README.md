# Training-Set Reduction Study: Mental Health Survey

Can a Random Forest be trained on far fewer rows without losing accuracy? This project benchmarks three ways of shrinking a 292K-row survey dataset (deduplication with count weights, a density-based k-NN + K-Means rule, and random sampling) and reports what each one costs and saves, using proper baselines and repeated train/test splits.

## Key findings

- **The data is much smaller than it looks.** After feature selection, 219,273 training rows collapse to just **155 distinct feature patterns**, and every test row matches a pattern seen in training. The task is close to a pattern lookup.
- **Deduplication with count weights is the clear win.** Replacing the training set with 199 weighted rows cut Random Forest training time by roughly **90–98%** (hardware-dependent) for about **1.5 points** of accuracy (75.7% to 74.1%, mean of 6 splits).
- **The density-based rule is not a reliable improvement.** Against random selection of the same number of patterns, it is unstable with Euclidean and Manhattan distance (gaps swing by about ±4 points and change sign between splits) and modestly better with cosine distance (+1.8 points, better in 6 of 6 splits).
- **Naive random row sampling is much worse than all of these** at the same row count (about 63–66% accuracy), because it throws away the information that the weights preserve.


## Dataset

| | |
|---|---|
| Rows | 292,364 |
| Features | 15 categorical survey answers (Gender, Country, Occupation, family_history, Growing_Stress, Mood_Swings, care_options, and others) |
| Target | `treatment` (Yes/No), roughly balanced (50.5% / 49.5%) |
| Missing values | `self_employed` only (5,202 rows), imputed with the most frequent value |
| Source | *(add dataset link here)* |

The `Timestamp` column is dropped.

## Method

1. **Split first.** A stratified 75/25 train/test split is made before anything is fitted.
2. **Preprocess (train only).** Most-frequent imputation and one-hot encoding (77 columns).
3. **Select features (train only).** Chi-square and mutual information (discrete) scores are normalised, summed, and the top 40 columns are kept.
4. **PCA (train only).** 95% of variance (9–10 components), used only to measure distances.
5. **Group identical rows.** Rows with identical selected features become one pattern, with a per-class count kept for each.
6. **Compare training sets**, each used to fit a 200-tree Random Forest and scored on the same untouched test set:

| Training set | How it is built |
|---|---|
| **All rows** | Baseline: every training row |
| **Dedup + weights** | One row per (pattern, class), weighted by its count |
| **Density rule** | On the patterns: score each by mean distance to its 5 nearest neighbours; keep the sparsest 20%; cluster the rest with K-Means and keep the 40% farthest from their centres. Tried with Euclidean, Manhattan and cosine distance |
| **Random unique patterns** | Control: random patterns, same count as the density rule |
| **Random raw rows** | Control: random individual rows, same count |

## Results

Single reference run (seed 42, all 292,364 rows):

| Training set | Rows used | Accuracy | ROC-AUC | Train time |
|---|---:|---:|---:|---:|
| All rows (baseline) | 219,273 | 75.5% | 0.844 | 12.95 s |
| Dedup + weights | 199 | 73.9% | 0.811 | 0.21 s |
| Density rule, Euclidean | 98 | 70.7% | 0.771 | 0.21 s |
| Density rule, Manhattan | 102 | 67.8% | 0.778 | 0.21 s |
| Density rule, cosine | 106 | 72.8% | 0.764 | 0.18 s |
| Random unique patterns (3 variants) | 106–109 | 71.1–72.1% | 0.773–0.777 | 0.20–0.26 s |
| Random raw rows (3 variants) | 98–106 | 63.5–66.4% | 0.688–0.718 | 0.19–0.21 s |

Repeated over **6 different train/test splits** (mean ± sd):

| | Accuracy |
|---|---|
| All rows | 75.67% ± 0.11 |
| Dedup + weights | 74.13% ± 0.22 (−1.54 points vs all rows) |

Density rule minus random unique patterns (same row count), in accuracy points:

| Distance | Mean gap | Range | Density rule better in |
|---|---:|---|---|
| Euclidean | −0.74 | −7.2 to +2.9 | 4 of 6 splits |
| Manhattan | −1.42 | −7.1 to +2.4 | 3 of 6 splits |
| Cosine | +1.78 | +1.3 to +2.4 | 6 of 6 splits |


## Limitations

- **One dataset, and a near-lookup task.** With only 155 patterns, this is a weak test of any sample-selection method. The conclusions may not transfer to datasets with continuous features or little duplication.
- **Weighted Random Forest is not equivalent to full training.** Bootstrap sampling ignores sample weights, which is part of why dedup loses about 1.5 points.
- **Timings depend on hardware.** Training-time savings were 91% on one machine and 98% on another. Every reduced variant takes about the same time because fitting 200 trees has a fixed cost, so compare timings against the full-data baseline, not against each other.
- **Results vary across environments.** On a second machine, dedup + weights scored 75.4% (versus about 74% here), possibly because of a library version difference. Run the scripts yourself and use your own numbers.
- **Leakage check.** An earlier version fitted preprocessing and feature selection on all rows before splitting. Fixing it produced identical results here, but the pipeline now fits everything on the training split only.


## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
```

1. Put `Mental Health Dataset.csv` somewhere and edit `CSV_PATH` at the top of both scripts.
2. Run the main experiment (a few minutes on all rows; set `SAMPLE_N = 40000` for a quick run):
   ```bash
   python sample_reduction_dedup_test.py
   ```
3. Run the robustness check (5–10 minutes on all rows):
   ```bash
   python sample_reduction_multisplit_check.py
   ```



## Tech stack

Python, pandas, NumPy, scikit-learn (Random Forest, K-Means, PCA, NearestNeighbors, chi-square and mutual-information feature selection), Matplotlib.



