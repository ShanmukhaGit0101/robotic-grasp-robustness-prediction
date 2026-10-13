# Part 2 — How to connect to Part 1 (handshake guide)

**For:** the person who owns the Part 2 notebook (`ds_training_selected_rf`).
**Goal:** make Part 2 use exactly the data, threshold and settings that Part 1 produced, and fail loudly if they ever differ. About 30 minutes of work, mostly deleting old lines.

---

## 1. The idea in one paragraph

Part 1 ends by writing a small file, `handoff.json`, next to the clean data. Part 2 starts by reading that file and **checking it** (checksum, row count, experiment count). Nothing about the data, the threshold, the extreme experiment or the feature groups is typed by hand in Part 2 any more. If Part 1 changes, Part 2 stops with a clear message instead of silently training on old data.

```
 Part 1 notebook                       Part 2 notebook
 ───────────────                       ───────────────
 data/clean.csv      ───────────────▶  loads this exact file
 reports/handoff.json ──────────────▶  reads threshold, extreme id, features, seed
                                        verifies checksum + row counts
                     ◀───────────────  reports/results_part2.json (at the end)
```

---

## 2. What you receive from Part 1

Ask the Part 1 owner for the folder `grasp-project/` (or just these files):

| File | Use |
|---|---|
| `data/clean.csv` | the training data (unscaled) |
| `data/dirty.csv`, `data/original.csv` | only for the Original / Dirty / Clean experiment |
| `reports/handoff.json` | the contract (see section 5) |

Put them somewhere Part 2 can reach with a **relative path** (for example the same Colab `/content/grasp-project/` folder, or a Drive folder you mount). Do not use `C:\Users\...` paths: they only work on one computer.

If the files are on Google Drive, mount it and set `HANDOFF` in the cell below to the Drive path.

---

## 3. Step by step

### Step 1 — Delete the old data-definition lines
Remove (or comment out) these from the top of the Part 2 notebook:

| Old line or cell | Why it goes |
|---|---|
| `file_path = r"C:\Users\lohit\grasping_data\shadow_robot_dataset.csv"` and the cells that load the original, dirty, clean and normalized files (cells 0–5) | Hard-coded and uses Sha's old clean file, which replaced real effort zeros |
| `STABILITY_THRESHOLD = 100.0` (cell 8) | Replaced by Part 1's data-driven threshold |
| `EXTREME_EXPERIMENT_ID = "3390f0ee..."` (cells 15, 31, 32, 35) | Read from the file; the id will be the same, but it is now verified |
| The duplicated cells (3 = 4, 22 = 25, 23 = 24, repeated linear baseline in 30) | Same code twice |

### Step 2 — Paste the handshake cell at the top
This is the file `part2_handshake_cell.py`. Paste it as the first code cell:

```python
import json, hashlib
from pathlib import Path
import pandas as pd, numpy as np

HANDOFF = Path("grasp-project/reports/handoff.json")        # in Colab: /content/grasp-project/reports/handoff.json
h = json.loads(HANDOFF.read_text())

clean_path = Path(h["clean_csv"])
assert hashlib.sha256(clean_path.read_bytes()).hexdigest() == h["sha256"], "clean.csv has changed since Part 1 - rerun Part 1 and republish handoff.json"
df_ml = pd.read_csv(clean_path)
df_ml.columns = df_ml.columns.str.strip()
assert len(df_ml) == h["rows"] and df_ml["experiment_number"].nunique() == h["experiments"], "row/experiment count differs from Part 1"

TARGET = TARGET_COLUMN = h["target"]                         # Part 2 used both names
ID_COLUMNS, RANDOM_STATE = h["id_columns"], h["seed"]
STABILITY_THRESHOLD, EXTREME_EXPERIMENT_ID = h["threshold"], h["extreme_experiment"]
POSITION_FEATURES, VELOCITY_FEATURES, EFFORT_FEATURES = (h["features"][k] for k in ("position", "velocity", "effort"))
FEATURE_COLUMNS = POSITION_FEATURES + VELOCITY_FEATURES + EFFORT_FEATURES
FEATURE_SETS = {"A_Position": POSITION_FEATURES, "B_Velocity": VELOCITY_FEATURES, "C_Effort": EFFORT_FEATURES,
                "D_Velocity_Effort": VELOCITY_FEATURES + EFFORT_FEATURES, "E_All": FEATURE_COLUMNS}
VELOCITY_EFFORT_FEATURES = VELOCITY_FEATURES + EFFORT_FEATURES
assert set(FEATURE_COLUMNS) <= set(df_ml.columns) and len(FEATURE_COLUMNS) == 27
assert EXTREME_EXPERIMENT_ID in set(df_ml["experiment_number"].str.strip()), "extreme experiment missing from the data"

X = df_ml[FEATURE_COLUMNS]
y_reg = df_ml[TARGET]
y_class = (y_reg >= STABILITY_THRESHOLD).astype(np.int8)    # 1 = Stable, 0 = Unstable, same rule as Part 1
print(f"Handshake OK: {len(df_ml):,} rows | threshold {STABILITY_THRESHOLD} ({h['threshold_method']}) | "
      f"{y_class.mean():.1%} of rows Stable | extreme experiment {EXTREME_EXPERIMENT_ID[:8]}...")
print("Part 1 screen to compare against:", h["part1_screen"]["regression_r2"])
```

### Step 3 — Variable names Part 2 already uses (nothing else to change)
The cell creates the same variable names the old cells created, so the rest of Part 2 keeps working:

| Variable | Old source | New source |
|---|---|---|
| `df_ml` | `pd.read_csv(clean_path)` | `clean.csv`, checksum-verified |
| `FEATURE_COLUMNS`, `X` | built from column names | `handoff.json` → `features` |
| `POSITION_FEATURES`, `VELOCITY_FEATURES`, `EFFORT_FEATURES`, `VELOCITY_EFFORT_FEATURES` | built by pattern matching | `handoff.json` → `features` |
| `TARGET_COLUMN`, `ID_COLUMNS`, `RANDOM_STATE` | typed | `handoff.json` |
| `y_reg`, `y_class`, `STABILITY_THRESHOLD` | typed (100.0) | threshold from `handoff.json` |
| `EXTREME_EXPERIMENT_ID` | typed | `handoff.json` |

### Step 4 — Keep your fold design, but rebuild it on the new data
Your split is good: an experiment-level 80 / 20 split with the extreme experiment forced into development, then `GroupKFold` with 5 folds. Keep the code, but **rerun it from scratch**, because the experiment-to-fold assignment depends on the data and the seed. Do not reuse the old saved folds (`Fold_*.joblib`).

Two small changes inside it:
- Use `RANDOM_STATE` from the handshake (it is 42, as before).
- Save only the fold assignment as a small CSV (`experiment_number, fold`) instead of joblib files in `Downloads`.

### Step 5 — Rerun everything top to bottom on a fresh kernel
Numbers will change slightly because the clean data is different (the old one replaced real effort zeros, and the threshold changes the classification target). Make sure no cell ends in an error (cell 16 of the old notebook ends in an `AssertionError`).

### Step 6 — Add what Part 2 is still missing
The handshake only fixes the inputs. The notebook should also cover:

| To add | How, using the handshake variables |
|---|---|
| **Classification** | Same folds, target `y_class`. Report accuracy, precision, recall, F1, ROC-AUC and the confusion matrix. Include a "predict the majority class" baseline. |
| **Feature-set experiments A–E** | Loop over `FEATURE_SETS`, same folds, one light model (LightGBM with about 200 trees), for regression and classification. |
| **Original vs Dirty vs Clean** | Load `original_csv` and `dirty_csv` from the handshake file, apply the same experiment-to-fold assignment, same model. Dirty has missing values, so put a median imputer in the pipeline. Evaluate on the **original** test rows to see the cost of bad data. |
| **Normalization experiment** | Unscaled vs Min-Max vs Z-score, scalers inside a `Pipeline` (fitted on training rows only), with linear or logistic regression. Tree models do not need scaling. |
| **Extreme experiment, both ways** | Report every metric with and without `EXTREME_EXPERIMENT_ID`. In the old results the CV RMSE is 24.5 while the test RMSE is 19.5, because only one fold contains the extreme experiment. |
| **Baseline and experiment-level metrics** | Add a dummy baseline and metrics averaged per experiment, not only per row. |
| **Model choice and size** | Random Forest, Extra Trees and LightGBM differ by about 0.003 R² against a fold standard deviation of 0.13, so choose by cost as well. Do not save a 5 GB `.joblib`; use a smaller forest or LightGBM, or store only metrics and predictions. |

### Step 7 — Write the results file at the end
So that the final report can be built from the two files, finish Part 2 with:

```python
results = {
    "model": "LightGBM",                                   # the chosen model
    "settings": {"n_estimators": 300, "num_leaves": 63},   # its settings
    "cv": {"MAE": ..., "RMSE": ..., "R2": ...},            # mean over folds
    "test": {"MAE": ..., "RMSE": ..., "R2": ...},          # frozen test set
    "test_excl_extreme": {"MAE": ..., "RMSE": ..., "R2": ...},
    "classification": {"Accuracy": ..., "Precision": ..., "Recall": ..., "F1": ..., "ROC_AUC": ...},
    "best_feature_set": "E_All",
    "data_quality": {"Original": ..., "Dirty": ..., "Clean": ...},   # test metric per training dataset
    "scaling": {"none": ..., "minmax": ..., "zscore": ...},
    "threshold": STABILITY_THRESHOLD, "handoff_sha256": h["sha256"],
}
Path("grasp-project/reports/results_part2.json").write_text(json.dumps(results, indent=2))
```
Including `handoff_sha256` records which Part 1 data these results belong to.

---

## 4. Checks that must pass before you share Part 2

| Check | Expected |
|---|---|
| First cell prints `Handshake OK` with the same row count, threshold and Stable share as Part 1 | yes |
| The notebook has no hard-coded `C:\...`, threshold or experiment id | none left (search for `C:\\`, `100.0`, `3390f0ee`) |
| `X.shape[1]` | 27 |
| Extreme experiment is in development, not in the frozen test set | yes |
| No experiment appears in two folds or in both development and test | yes (keep your audits, but print one line each) |
| The same fold assignment is used for every experiment in Step 6 | yes |
| Notebook runs top to bottom on a fresh kernel without errors | yes |
| `results_part2.json` exists and contains `handoff_sha256` equal to the one in `handoff.json` | yes |

---

## 5. What is inside `handoff.json`

| Key | Meaning |
|---|---|
| `clean_csv`, `dirty_csv`, `original_csv` | paths of the three data files |
| `rows`, `experiments` | counts for the clean file |
| `sha256` | checksum of `clean.csv` |
| `threshold`, `threshold_method`, `class_balance_stable_pct` | the Stable / Unstable rule (Stable = robustness ≥ threshold) and how it was chosen |
| `extreme_experiment`, `extreme_robustness` | the one extreme experiment (the id and its robustness value) |
| `seed`, `id_columns`, `target` | shared settings |
| `features` | `position`, `velocity`, `effort` lists (9 each) |
| `cleaning` | zero rule, columns where zeros were repaired or kept, the imputation method, noise columns and level |
| `part1_screen` | Part 1's classification metrics and regression R² for comparison |

---

## 6. If something goes wrong

| Message or symptom | What it means and what to do |
|---|---|
| `FileNotFoundError ... handoff.json` | The path in `HANDOFF` is wrong. Check where the `grasp-project` folder is on your machine or Drive. |
| `clean.csv has changed since Part 1` | Someone reran Part 1 or edited the file. Ask the Part 1 owner to rerun Part 1 completely and send both files again; do not edit the checksum. |
| `row/experiment count differs from Part 1` | You loaded the wrong CSV (original or dirty instead of clean). |
| `extreme experiment missing from the data` | The clean file was regenerated with a different seed or data. Same fix as the checksum message. |
| Metrics changed a lot from the old Part 2 | Expected to some extent: the clean data and the threshold are different. If R² for the regressors drops sharply, check that you are not comparing a number that included the extreme experiment with one that did not. |
| Part 1 changes later (for example the threshold) | Part 1 republishes both files, Part 2 reruns from the top. The checksum makes sure nobody forgets. |

---

## 7. Who does what

| Person | Task |
|---|---|
| Part 1 owner | Run the notebook once on the real data. Share `grasp-project/` (or `data/clean.csv`, `dirty.csv`, `original.csv` and `reports/handoff.json`). Do not change Part 1 afterwards without telling Part 2. |
| Part 2 owner | Steps 1–7 above. Report back with `results_part2.json`. |
| Both | Agree that neither side re-cleans or relabels the data, and that the threshold lives only in `handoff.json`. |
