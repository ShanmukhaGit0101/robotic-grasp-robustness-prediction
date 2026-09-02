# 🤖 Robotic Grasp Robustness Prediction

## Team Dataset Setup Guide

This guide explains how each team member can load the **Grasping Dataset** directly into their own Google Colab environment without sharing Kaggle API credentials.

---

## 📌 Dataset Information

**Dataset:** Grasping Dataset
**Source:** Kaggle
**Dataset ID:**

```text
ugocupcic/grasping-dataset
```

The dataset file downloaded for our project is:

```text
shadow_robot_dataset.csv
```

Our initial dataset inspection showed:

| Property                 |   Value |
| ------------------------ | ------: |
| Total Rows               | 992,641 |
| Total Columns            |      30 |
| Unique Experiments       |  53,937 |
| Approximate Memory Usage |  227 MB |
| Dataset Download Size    | ~183 MB |

> **Important:** The dataset contains approximately **53,937 unique grasp experiments**, but each experiment contains multiple measurement records. Therefore, the dataset has nearly **1 million rows**.

---

# 🔐 Step 1: Create a Kaggle Account

Each team member should have their own Kaggle account.

Go to Kaggle and create/login to your account.

**Do not share API tokens between team members.**

---

# 🔑 Step 2: Generate Your Kaggle API Token

In your Kaggle account:

1. Open **Settings**.
2. Find the **API** section.
3. Generate a new API Token.
4. Copy the generated token.

⚠️ **Keep your API token private.**

Do not:

* Send it in WhatsApp.
* Put it inside GitHub.
* Write it directly inside the shared Colab notebook.
* Share it with other team members.

---

# 🔒 Step 3: Add the Token to Google Colab Secrets

Open your Google Colab notebook.

On the left sidebar:

```text
🔑 Secrets
```

Create a new secret.

### Secret Name

```text
KAGGLE_API_TOKEN
```

### Secret Value

Paste **your own Kaggle API token**.

Enable notebook access if Colab asks for permission.

---

# 📓 Step 4: Install Kaggle in Colab

Run the following cell:

```python
!pip install -q kaggle
```

---

# 🔐 Step 5: Configure Kaggle Authentication

Run:

```python
from google.colab import userdata
import os

os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")

print("Kaggle authentication configured successfully!")
```

If this runs successfully, your Colab environment is ready to access Kaggle.

---

# 📥 Step 6: Download the Dataset

Run:

```python
!mkdir -p /content/data

!kaggle datasets download ugocupcic/grasping-dataset \
    -p /content/data \
    --unzip
```

The dataset will be downloaded directly into your Google Colab runtime.

### Dataset location:

```text
/content/data/
```

---

# 📂 Step 7: Check Downloaded Files

Run:

```python
!find /content/data -type f
```

Expected output:

```text
/content/data/shadow_robot_dataset.csv
```

---

# 📊 Step 8: Load the Dataset

Run:

```python
import pandas as pd
import numpy as np

file_path = "/content/data/shadow_robot_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)
```

Expected dataset shape:

```text
(992641, 30)
```

---

# 🔍 Step 9: Perform Initial Dataset Inspection

Run:

```python
df.head()
```

Check dataset information:

```python
df.info()
```

Check missing values:

```python
df.isna().sum()
```

Check the number of unique experiments:

```python
print(
    "Unique experiments:",
    df["experiment_number"].nunique()
)
```

Expected result:

```text
Unique experiments: 53937
```

---

# ⚠️ Important Dataset Structure

The dataset contains:

```text
992,641 Measurement Records
          │
          ▼
53,937 Unique Experiments
```

This means:

> **One experiment is represented by multiple measurement rows.**

Therefore, before beginning machine learning, we need to understand the dataset structure properly.

---

# 🔬 Current Team Task

Each team member should help investigate the following questions.

## 1. What does one row represent?

Determine whether each row represents:

```text
A complete grasp attempt

OR

One measurement during a grasp experiment
```

---

## 2. How many measurements are there per experiment?

Run:

```python
measurements_per_experiment = (
    df.groupby("experiment_number")
      .size()
)

print(measurements_per_experiment.describe())
```

Also visualize the distribution if required.

---

## 3. Is the robustness value constant for each experiment?

Run:

```python
robustness_per_experiment = (
    df.groupby("experiment_number")["robustness"]
      .nunique()
)

print(robustness_per_experiment.value_counts())
```

This will help determine whether robustness is:

```text
Experiment-Level Target

OR

Measurement-Level Target
```

---

## 4. Investigate Measurement Numbers

Run:

```python
df["measurement_number"].describe()
```

Also check:

```python
df.groupby("experiment_number")[
    "measurement_number"
].count().describe()
```

Determine whether:

> `measurement_number` represents sequential sensor measurements collected during a single grasp experiment.

---

# 🎯 Objective of This Investigation

Before performing:

```text
EDA
↓
Feature Engineering
↓
Regression
↓
Classification
```

We first need to decide:

```text
What should represent ONE machine learning observation?
```

Possible approaches:

### Approach A

```text
One Row
=
One ML Observation
```

### Approach B

```text
Multiple Measurements
        ↓
Aggregate by Experiment
        ↓
One Experiment
=
One ML Observation
```

We should **not make this decision until the dataset structure has been analyzed**.

---

# 🤝 Team Collaboration Rules

## Shared Through GitHub

Upload:

* Python code
* Jupyter notebooks
* Markdown documentation
* Visualizations
* Results
* README files

## Do NOT Upload to GitHub

```text
❌ Kaggle API Token
❌ KAGGLE_API_TOKEN
❌ kaggle.json
❌ access_token
❌ Large raw dataset
```

---

# 📁 Recommended Team Workflow

```text
                Kaggle
                   │
                   │
        Public Grasping Dataset
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     Member 1   Member 2   Member 3
        │          │          │
        ▼          ▼          ▼
    Google Colab Environments
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
                 GitHub
                   │
        Code + Notebooks + Results
```

---

# 🚀 Completion Checklist

Each team member should confirm:

```text
☐ Kaggle account created

☐ Personal API token generated

☐ Token added to Colab Secrets

☐ Dataset downloaded successfully

☐ Dataset loaded successfully

☐ Dataset shape verified

☐ experiment_number investigated

☐ measurement_number investigated

☐ robustness behaviour investigated
```

---

## 📌 Important Note

**Never share your personal Kaggle API token.**

Each team member should use their own token, while everyone uses the **same shared Colab notebook and GitHub repository**.



