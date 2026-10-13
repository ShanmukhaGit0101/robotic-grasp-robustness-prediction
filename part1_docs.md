# Part 1 — Documentation

**Project:** Predicting robotic grasp stability with machine learning (Shadow Robot grasping dataset)
**Notebook:** `Part1_Grasp_Stability.ipynb` (69 cells, about 450 lines of code)
**Scope:** data understanding, data quality (dirtying and cleaning), EDA and correlation, a first modelling screen, and the hand-over file for Part 2.

> **Status.** The notebook was tested end to end on a synthetic dataset with the same schema as the real one, and it ran without errors. It has **not** been run on the real Kaggle data yet. Every number quoted as "real" below comes from the team's earlier notebooks; numbers the notebook will print on the real data are described, not promised.

---
## ipunb link: https://colab.research.google.com/drive/1KZLKAyWXrhA290OZVnTD8SCnBKd0d_vu?usp=sharing
## 1. What Part 1 delivers

| Deliverable | File | Written by |
|---|---|---|
| Clean dataset (unscaled) | `data/clean.csv` | Stage 3 |
| Dirty dataset | `data/dirty.csv` | Stage 2 |
| Hand-over file for Part 2 | `reports/handoff.json` | end of the notebook |
| Data dictionary | `reports/data_dictionary.md` | Stage 1 |
| Baseline statistics | `reports/baseline_statistics.csv` | Stage 1 |
| Dirtying log (every modification) | `reports/dirty_data_log.csv` | Stage 2 |
| Cleaning validation (Original, Dirty, Clean) | `reports/cleaning_comparison.csv` | Stage 3 |
| Normalization comparison | `reports/normalization_comparison.csv` | Stage 3 |
| Correlation results | `reports/correlation_results.csv` | Stage 4 |
| Feature analysis and recommendation | `reports/feature_analysis.csv` | Stage 4 |
| Preliminary model results | `reports/preliminary_models.csv` | Stage 5 |
| Figures (PNG, 150 dpi) | `figures/*.png` | all stages |

The notebook creates the folder `grasp-project/` (in `/content/` on Colab) with `data/`, `reports/` and `figures/` inside it.

---

## 2. How to run it

1. **Colab:** add your own Kaggle token as a secret named `KAGGLE_API_TOKEN` (Secrets, key icon). Do not share tokens, and do not put them in the notebook or on GitHub.
2. **Local machine:** set the environment variable `GRASP_CSV` to the path of `shadow_robot_dataset.csv` instead.
3. Run all cells top to bottom. Expect tens of minutes on the full dataset (dirtying, cleaning and the random forest are the slow parts). The first run downloads about 183 MB.
4. Do not edit a cell and run only part of the notebook: stages read the previous stage's file from disk.

**Facts about the real data (from the team's earlier notebooks):** 992,641 rows, 30 columns, 53,937 experiments (1 to 30 rows each, median 15), 27 input features (9 positions, 9 velocities, 9 efforts), one target (`robustness`), two identifiers. The raw CSV headers contain stray spaces (for example `' robustness'`), which the notebook strips.

---

## 3. Notebook map

| Cells | Section | Purpose |
|---|---|---|
| 0–2 | Overview, setup, download | Imports, seed, plot style, folders, one-time download |
| 3–13 | **Stage 1** — Understanding | Structure, data dictionary, unique values, statistics, finger-wise view, baseline quality |
| 14–19 | **Stage 2** — Dirtying | Controlled corruption of a copy, log, before and after, noise study |
| 20–36 | **Stage 3** — Cleaning | Detect, repair, validate, scatter comparison, normalization study |
| 37–54 | **Stage 4** — EDA | Robustness, distributions, fingers, correlations, redundancy, threshold, feature table |
| 55–63 | **Stage 5** — First models | Logistic regression, regression models, importance |
| 64–68 | Hand-over and findings | Writes `handoff.json`, prints the findings, lists what Part 2 will do |

(Cell numbers count from 0 and are the same in Jupyter and Colab; the markdown headings in the notebook match these sections.)

---

## 4. Setup cell: what is fixed for the whole notebook

| Setting | Value | Why |
|---|---|---|
| `SEED` | 42 | Every random choice is seeded (dirtying, samples, splits, models). |
| Colours | blue = original, red = dirty, teal = clean, grey = background | A dataset version has the same colour in every plot. |
| Measurement colours | blue = position, amber = velocity, teal = effort | Same idea for the three sensor types. |
| Plot style | no top/right spines, light grid, constrained layout, 150 dpi | Consistent and report-ready; figures are also saved to `figures/`. |
| `ID_COLS`, `TARGET` | `experiment_number`, `measurement_number`; `robustness` | Used everywhere instead of retyping names. |

---

## 5. Stage 1 — Dataset understanding

**What the code does**

- Loads `original.csv`, strips the header spaces and builds the feature lists. Feature type, finger and joint are read from the column name, for example `H1_F3J2_pos` is finger 3, joint 2, position.
- Checks structure: rows per experiment, that rows are time-ordered inside every experiment, and that `robustness` is constant inside every experiment.
- Builds the **data dictionary** (also written as Markdown), with column, type, category, description and ML role.
- Unique-value analysis: the share of the most common value per column, to find identifier-like and almost-constant columns.
- Statistical profile for all 27 features and the target: min, max, mean, median, std, variance, Q1, Q3, range, IQR, skewness and the share of exact zeros. Printed highlights: highest and lowest std, largest range, most skewed feature.
- A bar chart of the **share of exact zeros per feature**, with the 1 % cut-off marked. This chart drives a later decision (section 7.2).
- Finger-wise comparison (median and IQR by finger and type) and the **baseline quality** table (missing, duplicates, infinite values, zeros, maximum robustness).

**What it found (real data, from the team's notebooks)**

- Original data: 0 missing values, 0 duplicate rows. Any such defect later was introduced by us.
- `robustness` is a per-experiment label repeated on every row. Rows of the same experiment are therefore not independent.
- Velocities have extreme tails (narrow IQR but ranges thousands of units wide). These are physical spikes, not errors.
- Several effort columns read exactly zero for a large share of rows (for example `H1_F1J2_eff`: 332,509 zeros, about a third of its rows).
- The three fingers are not interchangeable; they differ mostly in effort and in the J2 position.

---

## 6. Stage 2 — Artificial data dirtying

**Rule:** the original dataframe is never modified; all damage goes into a copy.

| Defect | Where | Amount | Implementation |
|---|---|---|---|
| Noise | the J2 joint of each finger (9 columns) | Gaussian, 5 % of the column's std | Added to non-zero readings only, so real zeros stay zero |
| Missing values | all 27 features | 0.1 % of rows per column | `NaN` |
| Artificial zeros | all 27 features | 0.1 % of rows per column | Only on cells that were non-zero |
| Outliers | all 27 features | 0.01 % of rows per column, at least 20 | Alternating `Q3 + 3·IQR` and `Q1 − 3·IQR` of the original |
| Duplicate rows | whole rows | 0.1 % of rows | Appended at the end of the file |

**Design points**

- A boolean mask `used` makes sure a cell is never corrupted twice, so the log is exact.
- Everything is vectorised with a seeded `numpy` generator, and every change is logged with row, column, original value, new value, error type and measurement type.
- The dirty file is written to disk and **read back in Stage 3**, so the stages are really connected.
- Outputs: `data/dirty.csv`, `reports/dirty_data_log.csv`, a summary table (defect by type, columns and share of data), a before and after table and a noise-level figure.

**Noise study.** One velocity channel is shown at 0, 5, 10 and 20 % noise. 5 % barely changes the shape; 20 % visibly widens it. Noise cannot be detected by a cleaning rule, so it stays in the clean data and is a robustness question for Part 2.

---

## 7. Stage 3 — Data cleaning

**Principle:** problems are found from the data itself. The log and the original are used only afterwards, to validate the result.

### 7.1 Missing values and duplicates
Counts missing cells per column and by type, then removes duplicate rows with `keep="first"`. Because duplicates were appended at the end, the index of the remaining rows still matches the original rows, which makes the later validation exact. Repeated `experiment_number` values are normal and are not duplicates.

### 7.2 Zeros: the central decision
> **Rule:** a zero is treated as an error only in columns where exact zeros are **rare** (below 1 % of rows). Where zeros are common, they are kept.

- The earlier cleaning replaced every zero in every feature with the column median (538,557 values), including real effort readings. That destroyed true "joint at rest" behaviour.
- With the rule, the position, velocity and low-zero effort columns are repaired and the natural-zero effort columns are left alone. On the real data the earlier counts suggest six effort columns (`H1_F1J2_eff`, `H1_F1J3_eff`, `H1_F3J2_eff`, `H1_F3J3_eff`, `H1_F2J3_eff`, `H1_F2J2_eff`, each 29–34 % zeros) are kept; the notebook prints the exact list.
- **Trade-off:** the 0.1 % of artificial zeros injected into the natural-zero columns cannot be told apart from real zeros and stay. The validation table reports this honestly (recall below 100 % for "Artificial Zero").

### 7.3 Outliers: artificial vs physical
- A plain 1.5·IQR rule would flag tens of thousands of valid velocity and effort spikes. The notebook prints how many more cells it would flag.
- Injected outliers have a fingerprint: **the same exact value repeated** outside the fences, while natural readings of a continuous sensor rarely repeat.
- Rule: a value outside the fences that appears between `REPEAT_MIN` and `REPEAT_MAX` times is an artificial outlier. `REPEAT_MIN = max(5, n // 50 000)`; `REPEAT_MAX = max(60, 0.0002 · n)`. Zero is excluded (handled by the zero rule). Values repeated *very* often are sensor saturation limits, which are valid.
- A safety assert stops the run if the rule would flag more than 1 % of any column.

### 7.4 Repair method: chosen by measurement
Cells to repair = missing + suspicious zeros + flagged outliers. Three methods are compared **against the true original values** of those cells (possible because we created the corruption), using the error divided by the feature's standard deviation:

1. global mean, 2. global median, 3. **linear interpolation along the time axis inside the same experiment** (falls back to the global median if an experiment has no valid neighbour).

The method with the lowest error is used. Interpolation is expected to win because a joint trajectory is smooth; the table the notebook prints is the evidence. Dropping rows is shown only as a cost (the share of rows that would be lost). This relies on rows being time-ordered inside each experiment, which Stage 1 checks.

### 7.5 Validation
- **Original, Dirty and Clean table:** rows, missing, duplicates, and mean, median, std, min and max of robustness, plus the mean absolute error on the corrupted cells (in standard-deviation units) for Dirty and Clean.
- **Detection table:** injected vs found vs false alarms vs missed, with recall and precision for each defect type.
- **Scatter comparison (section 3.5 of the notebook):** two features (a position and a velocity), three panels each. The *same* cells are shown as they should be (original), as the corruption made them (dirty) and as cleaning left them (clean), over a grey background sample. In the dirty panels the cells jump to zero or to the fence values; in the clean panels they return to the main cloud.

### 7.6 Normalization study
Min-Max and Z-score are compared on the clean data: both preserve the shape of every feature (skewness unchanged); Min-Max is dominated by extreme values (the central half of the data occupies a small part of [0, 1]). **Decision:** the clean file is saved **unscaled**; scaling is done inside model pipelines so scalers are fitted on training data only (no leakage). Z-score is preferred for scale-sensitive models; tree models are unaffected.

---

## 8. Stage 4 — EDA and correlation (on the clean data)

| Section | What it does | Key decision or reading |
|---|---|---|
| Robustness | Statistics at row level and experiment level, histogram, log-histogram, boxplot | Right-skewed, two groups, and one extreme experiment. The extreme experiment is **kept**. |
| Distributions | 3×3 histograms for each type (0.5–99.5 percentile window) and boxplots | Positions are compact, velocities heavy-tailed, efforts near zero with spikes. |
| Finger-wise | Spread (IQR) by joint and finger | Fingers behave differently; they are not interchangeable. |
| Correlations | Pearson and Spearman with robustness at row level, plus Spearman at experiment level | Spearman is the more trustworthy ranking: target and several features are heavy-tailed. Sign flips between row and experiment level are counted. |
| Scatter plots | Strongest and weakest feature of each type, chosen by |Spearman|, drawn as hexbin density plots | Checks that coefficients match what the data looks like. |
| Redundancy | Pairs with |r| ≥ 0.8 (Pearson or Spearman), Spearman heatmap (lower triangle, position, velocity and effort blocks) | Redundant features are **flagged, not removed**. |
| Threshold | See section 9 | Chosen from the data. |
| Feature table | Evidence label from max(|Spearman| row level, |Spearman| experiment level): strong ≥ 0.30, moderate ≥ 0.10, otherwise weak; redundant partners listed | Low correlation does not mean useless; tree models can use interactions. |

Heavy computations use a 200,000-row random sample; experiment-level statistics use the full data.

---

## 9. The Stable / Unstable threshold

**Problem.** The team used two different thresholds (EDA: median ≈ 94.8, a forced 50/50 split; classifier: 100). The percentiles (5 % ≈ 10, 25 % ≈ 15.4, 50 % ≈ 94.8, 75 % ≈ 118.4) suggest a low group and a main group, so 100 would cut through the main group.

**Method (in code).**
1. Estimate the density of robustness **per experiment** (not per row).
2. Find the peaks (prominence at least 5 % of the maximum). If there are two or more, take the two dominant modes and use the **valley** between them.
3. Otherwise fall back to the experiment median.
4. Round to the nearest 5 for readability.
5. Compare against alternatives (median, 100, the first quartile) for class balance at experiment and row level, and plot the density with all candidate thresholds.

The result is stored in `THRESHOLD` and written to `handoff.json`; it is used everywhere afterwards. Stable means `robustness >= THRESHOLD`.

---

## 10. Stage 5 — First modelling screen

A deliberately **quick and preliminary** comparison, so Part 2 knows which model families are worth taking forward.

| Item | Choice |
|---|---|
| Data | clean data, a random subset of up to 5,000 experiments (all rows of each) |
| Split | one 80 / 20 split **by experiment** (`GroupShuffleSplit`), with an assert that no experiment is on both sides |
| Classification | logistic regression with a standard scaler. Metrics: accuracy, precision, recall, F1, ROC-AUC, plus the majority-class accuracy for context; confusion matrix (counts and row-normalised) |
| Regression | dummy (mean), linear regression (scaled), decision tree (depth 12), random forest (100 trees, depth 15). Metrics: MAE, RMSE, R², and RMSE excluding the extreme experiment |
| Plots | one point per test experiment (mean prediction vs actual) with shared axes and a residual panel; the extreme experiment is excluded from the R² shown; importance by feature and by measurement type |

**Why one point per experiment?** Rows of an experiment share one actual value, so a row-level residual plot draws diagonal streaks (a fixed actual minus a varying prediction is a straight line with slope −1). That was the strange pattern in the team's earlier residual plot.

Outputs: `reports/preliminary_models.csv` and the figures `05_*.png`.

---

## 11. The hand-over file (`reports/handoff.json`)

Written at the end of the notebook. Part 2 reads it and stops if the clean file has changed.

| Key | Content |
|---|---|
| `clean_csv`, `dirty_csv`, `original_csv` | absolute paths of the three data files |
| `rows`, `experiments` | row and experiment counts of the clean file |
| `sha256` | checksum of `clean.csv` |
| `threshold`, `threshold_method`, `class_balance_stable_pct` | the Stable/Unstable rule |
| `extreme_experiment`, `extreme_robustness` | the single extreme experiment |
| `seed`, `id_columns`, `target` | shared settings |
| `features` | the 27 features grouped as `position`, `velocity`, `effort` |
| `cleaning` | zero cut-off, columns where zeros were repaired and kept, chosen imputer, noise columns and level |
| `part1_screen` | classification metrics and regression R² of the preliminary models |

See `PART2_HANDSHAKE_GUIDE.md` for how Part 2 uses it.

---

## 12. Decision log

| # | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 1 | One notebook with connected stages; each stage reads the previous file from disk | Separate notebooks with manual file passing | The earlier cleaning ran on a different file (276,996 rows) from the one dirtying produced (993,633). Printing and asserting `shape` prevents this. |
| 2 | Split and CV by `experiment_number` | Random row split | All rows of an experiment share one target; row splits leak and overstate performance. |
| 3 | Zeros are errors only where they are rare (< 1 %) | Replace all zeros (earlier approach); replace none | Real effort zeros are about a third of some columns. Replacing them removes true behaviour. |
| 4 | Outliers found by repeated-value signature | Plain 1.5·IQR; drop rows | IQR flags tens of thousands of valid physical spikes. |
| 5 | Imputation by in-experiment interpolation, chosen by measured error | Mean or median (earlier approach); drop rows | Trajectories are smooth; the choice is backed by a table against the truth. |
| 6 | Clean file stored unscaled; scalers in pipelines | Scale the whole dataset (earlier approach) | Fitting on all data leaks test information. |
| 7 | Z-score for scale-sensitive models | Min-Max | Heavy tails squeeze most data into a tiny part of [0, 1]. |
| 8 | Noise added to non-zero readings only | Noise on every reading | Keeps real zeros as zeros, matching the zero policy. |
| 9 | Noise left in the clean data | Try to filter it | There is nothing to detect; it is a model-robustness question for Part 2. |
| 10 | Threshold from the density valley | Median (50/50 by construction); 100 (arbitrary) | Based on the data's own structure; also compared and plotted against the alternatives. |
| 11 | Extreme experiment kept; metrics reported with and without it | Delete it | It is genuine simulator output. Deleting a real observation needs a stronger reason than convenience. |
| 12 | Spearman as the main ranking, Pearson alongside | Pearson only | Heavy tails distort Pearson. |
| 13 | Redundant and weakly correlated features flagged, not removed | Drop by threshold | Trees can use interactions; the decision is left to Part 2's experiments. |
| 14 | Part 1 models are only a screen (subset, one split) | Full CV and tuning here | That is Part 2's job; Part 1 stays fast and simple. |
| 15 | One-point-per-experiment diagnostics | Row-level scatter and residuals | Removes the streak artefact and avoids over-weighting long experiments. |
| 16 | Handshake through `handoff.json` with a checksum | Retyping values in Part 2 | Prevents silent mismatch between the parts. |

---

## 13. Corrections compared with the earlier work

| Earlier issue | What Part 1 does now |
|---|---|
| Cleaning read a different dirty file | Reads the exact file written in Stage 2 and asserts its shape |
| Every zero replaced by the median | Zero rule by column (section 7.2) |
| EDA and models used the original data | Everything downstream uses the cleaned data |
| Two thresholds (94.8 and 100) | One data-driven threshold, passed on in `handoff.json` |
| Min-Max fitted on the full data | Clean file unscaled; scalers inside pipelines |
| No noise step, no before/after table | Noise step, summary and before/after tables |
| Outlier counts in the log disagreed | Exact masks; log built once per run |
| Mean imputation only, outliers never treated | Method compared against the truth; outliers repaired |
| No validation of cleaning | Original, Dirty and Clean table and a detection table |
| Streaky residual plot | One point per experiment, shared axes |
| Hard-coded paths, repeated downloads, `files.download` pop-ups | One setup cell and relative paths |

---

## 14. Limitations

- The data is simulated (ROS/Gazebo); real-robot noise and dynamics will differ.
- Artificial zeros inside the natural-zero effort columns cannot be separated from real zeros (a deliberate trade-off).
- The outlier detector is tuned to a repeated-value signature. It suits our controlled corruption; real-world outliers may need other rules. Always check the "found / missed" table on the real data.
- The Stage 5 models use a single split on a subset, so the numbers are indicative only. Part 2 provides the proper evaluation.
- Single time steps are used as samples; the temporal structure of a grasp is ignored.
- Noise is not removed.

---

## 15. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Download fails | The Kaggle secret is missing or named differently. It must be `KAGGLE_API_TOKEN`. Or set `GRASP_CSV` to a local copy. |
| `AssertionError: the file on disk is not the dirty file created in Stage 2` | Stage 2 was not run before Stage 3 in this session. Run top to bottom. |
| `AssertionError: the rule flags more than 1 % of a column` | The outlier rule needs review on this data; inspect the column's repeated values. |
| Out-of-memory in Colab | Restart the runtime and run top to bottom; the notebook frees large objects after Stage 3. |
| Part 2 says the checksum changed | `clean.csv` was regenerated after `handoff.json` was written. Rerun Part 1 completely and share both files again. |
