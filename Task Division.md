# Predicting Grasp Stability in Robotic Hands

## Team Task Division

This project uses the **Shadow Robot Smart Grasping Dataset** to study grasp robustness and build a machine-learning model for predicting grasp stability.

The project follows the Data Science workflow covered in the course:

**Dataset Understanding → Data Dirtying → Data Cleaning → EDA & Statistical Analysis → ML Preparation & Modeling → Evaluation**

The work is divided among five team members so that each member has a clearly defined and substantial responsibility.

---

## Team Members

| Member      | Responsibility                                     |
| ----------- | -------------------------------------------------- |
| **Gow**    | Dataset Understanding & Statistical Profiling      |
| **Kranthi** | Artificial Data Dirtying / Data Quality Simulation |
| **Sha**     | Data Cleaning, Outlier Treatment & Normalization   |
| **Loh**     | ML Preparation, Modeling & Evaluation              |
| **Kish**     | EDA, Correlation & Feature/Target Analysis         |

---

# 1. Gow — Dataset Understanding & Statistical Profiling

### Objective

Understand the original Shadow Robot dataset completely and establish its **baseline condition before any artificial data corruption is introduced**.

### Work to be completed

#### 1. Dataset Structure

Study and document:

* Total number of rows
* Total number of columns
* Column names
* Data types
* Numerical and non-numerical columns
* Identifier columns
* Input features
* Target variable

#### 2. Complete Data Dictionary

Document every column in the dataset.

The robotic measurements should be organized according to:

* Finger
* Joint
* Position
* Velocity
* Effort

The target variable, `ROBUSTNESS`, must also be explained.

For every column, record:

| Column                | Type                 | Category | Description           | ML Role |
| --------------------- | -------------------- | -------- | --------------------- | ------- |
| Experiment identifier | Numerical/Identifier | ID       | Identifies experiment | Exclude |
| Joint position        | Numerical            | Position | Joint state           | Input   |
| Joint velocity        | Numerical            | Velocity | Joint movement        | Input   |
| Joint effort          | Numerical            | Effort   | Joint effort/torque   | Input   |
| ROBUSTNESS            | Numerical            | Target   | Grasp robustness      | Target  |

Use the **actual column names from the CSV** when preparing the final table.

#### 3. Unique-Value Analysis

For each column, investigate:

* Number of unique values
* Most frequent values where meaningful
* Whether a column behaves like an identifier
* Whether any column has almost constant values

Use `nunique()` and `value_counts()` where appropriate.

#### 4. Statistical Profiling

For all 27 robotic features and `ROBUSTNESS`, calculate and analyze:

* Minimum
* Maximum
* Mean
* Median
* Standard deviation
* Variance
* Q1
* Q3
* Range
* IQR

Do not only generate the numbers.

Identify:

* Highest variation
* Lowest variation
* Largest range
* Features with unusual distributions
* Differences between position, velocity and effort

#### 5. Finger-wise Analysis

Compare the three fingers.

Analyze:

* Position statistics
* Velocity statistics
* Effort statistics

Determine whether the three fingers have similar or different statistical behavior.

#### 6. Original Data Quality

Before the dataset is made dirty, establish:

* Number of missing values
* Number of duplicate rows
* Data types
* Suspicious values
* Extreme values

This becomes the **baseline against which the dirty and cleaned datasets will later be compared**.

### Deliverables

```text
01_dataset_investigation.ipynb
data_dictionary.md
baseline_statistics.csv
```

The notebook should contain substantial analysis rather than only `head()`, `shape()` and `describe()`.

---

# 2. Kranthi — Artificial Data Dirtying / Data Quality Simulation

### Objective

Take the original dataset and intentionally introduce **controlled and realistic data-quality problems**.

The purpose is to simulate the type of problems that can occur in real-world robotic/sensor datasets.

### Important Rule

The original dataset must never be modified directly.

Create a separate copy:

**Original Dataset → Dirty Dataset**

---

## Work to be completed

### 1. Missing Values

Introduce controlled missing values into multiple types of robotic measurements:

* Joint position
* Joint velocity
* Joint effort

Do not place all missing values in one column.

Record:

* Columns affected
* Number of missing values
* Percentage of affected data

### 2. Duplicate Rows

Introduce a controlled number of duplicate records.

Record:

* Number of rows before duplication
* Number of rows after duplication
* Number of duplicate records introduced
* Percentage of duplicated records

Be careful to distinguish **actual duplicate rows** from repeated experiment identifiers.

### 3. Artificial Outliers

Introduce abnormal values into selected:

* Position features
* Velocity features
* Effort features

The values should be abnormal relative to the normal distribution rather than simply inserting meaningless extreme numbers.

The objective is to create outliers that the cleaning stage should be able to detect.

### 4. Artificial Noise

Use controlled random noise on selected sensor measurements.

Investigate the effect of different noise levels.

Compare:

* Original measurement distribution
* Noisy measurement distribution

### 5. Reproducibility

Use a fixed random seed so that the same dirty dataset can be reproduced.

The dirtying process must therefore be documented rather than manually modifying individual cells.

### 6. Dirtying Log

Maintain a complete record of every modification.

Example:

| Problem        | Measurement Type | Columns Affected | Amount Introduced |
| -------------- | ---------------- | ---------------- | ----------------: |
| Missing values | Position         | Selected columns |                 X |
| Missing values | Velocity         | Selected columns |                 X |
| Missing values | Effort           | Selected columns |                 X |
| Duplicates     | All              | Full rows        |                 X |
| Outliers       | Position         | Selected columns |                 X |
| Outliers       | Velocity         | Selected columns |                 X |
| Outliers       | Effort           | Selected columns |                 X |
| Noise          | Sensor features  | Selected columns |                 X |

### 7. Before/After Comparison

Compare the original and dirty datasets using:

* Dataset size
* Missing values
* Duplicate count
* Mean
* Median
* Standard deviation
* Minimum
* Maximum

### Deliverables

```text
02_data_dirtying.ipynb
grasping_dataset_dirty.csv
dirty_data_log.csv
```

---

# 3. Sha — Data Cleaning, Outlier Treatment & Normalization

### Objective

Take the **dirty dataset** and develop a systematic process to recover a clean dataset.

Sha should perform the cleaning **without simply referring to the dirtying log**.

The cleaning process should identify the problems from the data itself.

---

## Work to be completed

### 1. Missing-Value Detection

For every column:

* Count missing values
* Calculate missing-value percentage
* Identify which measurement types are most affected

Compare the missing-data situation before and after cleaning.

### 2. Missing-Value Treatment

Compare different approaches covered in the course:

* Removing missing rows
* Mean imputation
* Median imputation

For selected features, compare how each method changes:

* Mean
* Median
* Standard deviation
* Minimum
* Maximum

Select the most appropriate method and justify the choice.

### 3. Duplicate Detection

Identify:

* Number of duplicate rows
* Percentage of duplicates
* Duplicate records

Remove the duplicates and verify the resulting dataset.

### 4. Outlier Detection

Use the course methods:

* Five-number summary
* Percentiles
* Q1
* Q3
* IQR
* Boxplots

Perform this analysis on multiple robotic features covering:

* Position
* Velocity
* Effort
* `ROBUSTNESS`

### 5. Outlier Treatment

Determine which abnormal values are likely to be artificially introduced.

Compare:

**Original → Dirty → Clean**

and determine whether the cleaning procedure successfully removes the artificial corruption without unnecessarily removing valid observations.

### 6. Cleaning Validation

Create a comparison table:

| Metric             | Original | Dirty | Clean |
| ------------------ | -------: | ----: | ----: |
| Rows               |          |       |       |
| Missing values     |          |       |       |
| Duplicate rows     |          |       |       |
| Mean robustness    |          |       |       |
| Median robustness  |          |       |       |
| Standard deviation |          |       |       |
| Minimum            |          |       |       |
| Maximum            |          |       |       |

This is an important part of the project because it demonstrates whether the cleaning process actually worked.

### 7. Normalization

Apply both:

#### Min-Max Normalization

Study:

* Original range
* Normalized range
* Effect on the feature distribution

#### Z-score Normalization

Study:

* Original mean/std
* Normalized mean/std
* Effect on the feature distribution

Compare the two approaches and determine which is more appropriate for the ML stage.

### Deliverables

```text
03_data_cleaning.ipynb
clean_dataset.csv
cleaning_comparison.csv
normalization_comparison.csv
```

---

# 4. Loh — Machine Learning Preparation, Modeling & Evaluation

### Objective

Use the final cleaned dataset to develop the **machine-learning portion of the project**.

This section converts the cleaned robotic data into prediction problems.

---

## Work to be completed

### 1. Define ML Inputs

The candidate input features are the 27 robotic measurements:

* Joint positions
* Joint velocities
* Joint efforts

The identifier columns should be investigated and excluded from the ML input if they do not represent physical information.

### 2. Define Regression Target

Use:

`ROBUSTNESS`

as the continuous target.

The regression problem is:

**Joint measurements → Predicted grasp robustness**

### 3. Define Classification Target

Create:

**Stable / Unstable**

from the robustness score using the threshold justified by the team.

The team should document the reason for selecting the threshold.

### 4. Feature-Set Experiments

Compare different feature groups:

#### Experiment A

Position only

#### Experiment B

Velocity only

#### Experiment C

Effort only

#### Experiment D

Velocity + Effort

#### Experiment E

Position + Velocity + Effort

The purpose is to determine which types of robotic measurements are most useful for prediction.

### 5. Regression Model

Build the baseline regression model taught/required in the course.

Predict:

`ROBUSTNESS`

Evaluate using:

* MAE
* RMSE
* R²

### 6. Classification Model

Build the baseline classification model.

Predict:

`Stable / Unstable`

Evaluate using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 7. Data Quality Experiment

Compare ML performance using:

**Original dataset**

vs.

**Dirty dataset**

vs.

**Cleaned dataset**

This demonstrates the effect of data quality on machine-learning performance.

### 8. Normalization Experiment

Compare model performance using:

* Unnormalized features
* Min-Max normalized features
* Z-score normalized features

### 9. Final Model Comparison

Create a table:

| Dataset  | Feature Set       | Model          | Performance |
| -------- | ----------------- | -------------- | ----------- |
| Original | All               | Regression     |             |
| Dirty    | All               | Regression     |             |
| Clean    | All               | Regression     |             |
| Clean    | Velocity + Effort | Regression     |             |
| Clean    | All               | Classification |             |
| Clean    | Velocity + Effort | Classification |             |

Identify the best-performing approach.

### Deliverables

```text
04_ml_modeling.ipynb
model_results.csv
model_comparison.csv
```

---

# 5. Kish — EDA, Correlation & Feature/Target Analysis

### Objective

Perform detailed **Exploratory Data Analysis** on the cleaned dataset and determine how the robotic measurements relate to grasp robustness.

This section provides the statistical reasoning behind the features eventually used by the ML model.

---

## Work to be completed

### 1. Univariate Analysis

Analyze the distributions of:

* Joint positions
* Joint velocities
* Joint efforts
* `ROBUSTNESS`

Use appropriate:

* Histograms
* Boxplots
* Five-number summaries

Do not generate graphs without interpretation.

For each major group, identify:

* Central tendency
* Spread
* Skewness
* Potential unusual values

### 2. Finger-wise Analysis

Compare:

**Finger 1 vs Finger 2 vs Finger 3**

for:

* Position
* Velocity
* Effort

Determine whether the fingers exhibit similar or different behavior.

### 3. Robustness Analysis

Study `ROBUSTNESS` in detail.

Determine:

* Minimum
* Q1
* Median
* Q3
* Maximum
* Mean
* Standard deviation
* Distribution shape
* Presence of extreme values

Create a robustness histogram and boxplot.

### 4. Feature vs Robustness Analysis

Create scatter plots between robustness and selected robotic features.

Select features based on statistical relevance rather than randomly.

Investigate:

* Strong positive relationship
* Strong negative relationship
* Weak relationship

Do this for position, velocity and effort features.

### 5. Pearson Correlation

Calculate Pearson correlation between:

**Every robotic input feature and `ROBUSTNESS`.**

Rank the results:

* Top positive correlations
* Top negative correlations
* Weakest correlations

Create a table of the results.

### 6. Spearman Correlation

Perform the same analysis using Spearman correlation.

Compare:

**Pearson vs Spearman**

for the important features.

Investigate cases where Pearson and Spearman produce noticeably different results.

### 7. Feature-to-Feature Correlation

Study relationships among the 27 input variables.

Investigate:

* Position vs position
* Velocity vs velocity
* Effort vs effort
* Position vs velocity
* Position vs effort
* Velocity vs effort

Identify potentially redundant features.

### 8. Correlation Visualization

Create a correlation matrix/heatmap covering the relevant numerical features.

Use it to identify:

* Strong correlations
* Weak correlations
* Redundant variables
* Features related to robustness

### 9. Stable vs Unstable Analysis

After the team agrees on the robustness threshold:

Create the categorical target:

**Stable / Unstable**

Then calculate:

* Number of stable samples
* Number of unstable samples
* Percentage of each class

Visualize the class distribution using a bar chart.

### 10. Feature Selection Recommendation

Based on:

* Pearson correlation
* Spearman correlation
* Scatter plots
* Feature distributions
* Physical interpretation

prepare a recommended list of important features for the ML stage.

Do **not** remove features purely because their correlation is low; the final decision should be made together with the ML analysis.

### Deliverables

```text
05_eda_correlation.ipynb
correlation_results.csv
feature_analysis.csv
```

---

# Final Team Workflow

The five sections connect as follows:

```text
                         SHADOW ROBOT DATASET
                                  │
                                  ▼
                  ┌─────────────────────────────┐
                  │ GOW                         │
                  │ Dataset Understanding       │
                  │ Statistical Profiling       │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ KRANTHI                      │
                  │ Artificial Data Dirtying     │
                  │ Missing / Duplicate /       │
                  │ Outlier / Noise              │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ SHA                          │
                  │ Data Cleaning                │
                  │ Outlier Treatment            │
                  │ Normalization                │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ KISH                         │
                  │ EDA & Correlation            │
                  │ Feature/Target Analysis      │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │ LOH                          │
                  │ ML Preparation               │
                  │ Regression & Classification  │
                  │ Model Evaluation             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                       FINAL PROJECT RESULTS
```

---

# Shared Rules for All Members

### 1. Use the same dataset version

All members must work from the same original dataset and maintain consistent column names.

### 2. Do not overwrite datasets

Maintain separate files:

```text
data/
├── original/
│   └── grasping_dataset.csv
│
├── dirty/
│   └── grasping_dataset_dirty.csv
│
└── processed/
    └── grasping_dataset_clean.csv
```

### 3. Record every modification

Any change made to the dataset must be documented.

### 4. Don't generate graphs without analysis

Every important graph should answer a question.

### 5. Don't blindly remove data

Any row, feature or outlier removed must have a reason.

### 6. Keep the work reproducible

Use fixed random seeds wherever randomness is involved.

### 7. Use actual dataset values

Do not assume the number of rows, columns, missing values or outliers. All final numbers must come from the actual CSV.

### 8. Don't duplicate work

Each member owns their section, but the final results must be integrated into one pipeline.

---

# GitHub Repository Structure

```text
grasp-stability-project/
│
├── README.md
│
├── data/
│   ├── original/
│   │   └── grasping_dataset.csv
│   │
│   ├── dirty/
│   │   └── grasping_dataset_dirty.csv
│   │
│   └── processed/
│       └── grasping_dataset_clean.csv
│
├── notebooks/
│   ├── 01_dataset_investigation.ipynb
│   ├── 02_data_dirtying.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_eda_correlation.ipynb
│   └── 05_ml_modeling.ipynb
│
├── results/
│   ├── statistics/
│   ├── cleaning/
│   ├── eda/
│   └── models/
│
├── reports/
│   ├── data_dictionary.md
│   ├── dirty_data_log.csv
│   ├── cleaning_comparison.csv
│   └── model_comparison.csv
│
└── requirements.txt
```

---

# Responsibility Summary

| Member      | Owns                     | Main Question                                                          |
| ----------- | ------------------------ | ---------------------------------------------------------------------- |
| **Gow**    | Dataset + Statistics     | **What is in our dataset?**                                            |
| **Kranthi** | Dirtying                 | **How can we simulate realistic data problems?**                       |
| **Sha**     | Cleaning + Normalization | **How can we detect and fix those problems?**                          |
| **Kish**     | EDA + Correlation        | **What patterns and relationships exist in the cleaned data?**         |
| **Loh**     | ML + Evaluation          | **Can we use the cleaned data to predict grasp robustness/stability?** |

## Final Project Question

> **Can a deliberately corrupted Shadow Robot grasping dataset be systematically cleaned and analyzed to build a machine-learning model capable of predicting grasp robustness and classifying grasps as stable or unstable?**
