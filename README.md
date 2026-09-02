# 🤖 Predicting Robotic Grasp Stability Using Machine Learning

## 📌 Project Overview

Robotic manipulation requires reliable and stable grasping. A robotic grasp may initially appear successful but become unstable when the robot or object experiences movement or external disturbances.

This project applies **Data Science and Machine Learning techniques** to robotic sensor data in order to investigate whether joint measurements can be used to predict the robustness and stability of a robotic grasp.

The dataset contains simulated grasp attempts from a **3-fingered robotic hand grasping an object in a ROS and Gazebo simulation environment**. Each grasp attempt includes numerical sensor measurements such as joint position, velocity, and effort or torque.

The project treats robotic grasp analysis as a **tabular machine learning problem**.

---

## 🎯 Problem Statement

> Can robotic joint angle, velocity, and torque measurements be used to predict the robustness and stability of a robotic grasp using machine learning?

---

## 🎯 Objectives

* Perform data cleaning and preprocessing on robotic grasp sensor data.
* Conduct exploratory data analysis to understand sensor behaviour and grasp robustness.
* Develop regression models to predict continuous grasp robustness.
* Develop classification models to identify stable and unstable grasps.
* Compare machine learning model performance.
* Identify important sensor measurements influencing grasp stability.

---

## 🧠 Machine Learning Tasks

### 1. Regression

Predict a continuous grasp robustness score using robotic joint sensor measurements.

**Input Features:**

* Joint positions
* Joint velocities
* Joint effort / torque

**Output:**

* Grasp robustness score

**Evaluation Metrics:**

* MAE
* RMSE
* R² Score

---

### 2. Classification

Convert grasp robustness into meaningful stability categories.

```text
Robustness Score
       ↓
Threshold Selection
       ↓
Stable / Unstable
```

**Evaluation Metrics:**

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## ⚙️ Project Workflow

```text
Robotic Grasp Dataset
         ↓
Data Understanding
         ↓
Data Cleaning
         ↓
Exploratory Data Analysis
         ↓
Feature Engineering
         ↓
Regression Modeling
         ↓
Robustness Prediction
         ↓
Classification Modeling
         ↓
Stable / Unstable Prediction
         ↓
Model Evaluation
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📊 Dataset

The project uses a simulated robotic grasping dataset containing approximately **54,000 grasp attempts**.

Each observation represents a robotic grasp attempt and contains numerical measurements related to:

* Joint position
* Joint velocity
* Joint effort
* Grasp robustness

---

## 📂 Repository Structure

```text
robotic-grasp-stability-ml/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── presentation/
├── docs/
├── README.md
└── requirements.txt
```

---

## 🚀 Future Work

Potential extensions include:

* Explainable AI using SHAP
* Feature importance analysis
* Sensor reduction
* Advanced machine learning models
* Neural networks
* Noise robustness analysis
* Physics-informed feature engineering

---

## 👨‍💻 Author

**Shanmukha Vinayak M**

B.Tech Automation and Robotics Engineering

---

## 📌 Project Status

🚧 **Currently under development**

The project is being developed as a Data Science course project with potential future extension into robotics research.
