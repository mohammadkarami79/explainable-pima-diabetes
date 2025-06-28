# Interpretable Pima Indian Diabetes Classification

This repository contains the implementation and analysis for Question 1 of Assignment 2 from the "Trusted AI" course at the University of Tehran's Department of Electrical Engineering and Computer Science. The objective is to build a neural network classifier for the Pima Indian Diabetes dataset and apply model‑agnostic interpretability techniques to understand feature importance and model behavior.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [Data Preprocessing](#data-preprocessing)
* [Model Training](#model-training)
* [Model Interpretation](#model-interpretation)

  * [LIME](#lime)
  * [SHAP](#shap)
  * [Neural Additive Model (NAM)](#neural-additive-model-nam)
* [Results](#results)
* [Usage](#usage)
* [Requirements](#requirements)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Overview

The Pima Indian Diabetes dataset consists of medical diagnostic measurements for female Pima Indian patients and a binary outcome indicating the presence of diabetes. In this project, we:

1. Load and explore the dataset.
2. Preprocess the data (imputation, scaling, train/validation/test split).
3. Train a neural network classifier using PyTorch.
4. Evaluate classification performance.
5. Apply LIME, SHAP, and Neural Additive Model (NAM) interpretability methods to investigate feature contributions.

## Dataset

The dataset is sourced from the UCI Machine Learning Repository. It includes 768 instances and 8 features:

* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`
* `Outcome` (0: healthy, 1: diabetes)

A detailed statistical summary and data exploration are provided in the Jupyter notebook.

## Project Structure

```
.
├── data/
│   └── pima_diabetes.csv
├── notebooks/
│   └── Q1_interpretability.ipynb
├── models/
│   └── diabetes_nn.pth
├── README.md
└── requirements.txt
```

* `data/` contains the raw dataset.
* `notebooks/` includes the Jupyter notebook for Q1.
* `models/` stores trained model checkpoints.
* `README.md` (this file).
* `requirements.txt` lists Python package dependencies.

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/interpretable-pima-diabetes-classification.git
   cd interpretable-pima-diabetes-classification
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

## Data Preprocessing

The notebook performs:

* Missing value imputation (zero values replaced or imputed).
* Feature scaling using `StandardScaler`.
* Stratified split: 70% train, 20% validation, and 10% test sets.

## Model Training

A feedforward neural network is trained in PyTorch with:

* Binary Cross-Entropy Loss with logits.
* Adam optimizer.
* 100 epochs training loop with training and validation loss/accuracy tracking.

Performance metrics (accuracy, confusion matrix) are evaluated on the test set.

## Model Interpretation

### LIME

Local interpretable model‑agnostic explanations are generated for random test samples to highlight feature contributions.

### SHAP

SHAP values are computed to estimate global and local feature importance across test samples.

### Neural Additive Model (NAM)

A NAM is trained and visualized to show feature‑wise learned functions, providing a glass‑box interpretability.

## Results

* Test accuracy: **0.XX**

* Confusion matrix:

  ```
  [[ TN  FP ]
   [ FN  TP ]]
  ```

* Feature importance rankings: Glucose > BMI > Age > ...

Detailed plots and explanation are available in the notebook.

## Usage

1. Run the notebook `notebooks/Q1_interpretability.ipynb` end-to-end.
2. Modify configurations (e.g., number of epochs) directly in the notebook.
3. Use saved model from `models/diabetes_nn.pth` for inference.

## Requirements

* Python 3.8+
* pandas
* numpy
* scikit-learn
* matplotlib
* torch
* lime
* shap
* neural-additive-models

Install via:

```bash
pip install pandas numpy scikit-learn matplotlib torch lime shap neural-additive-models
```
