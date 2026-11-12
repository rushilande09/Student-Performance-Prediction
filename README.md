# Student Performance Prediction

A machine learning project that predicts students' final grades (`G3`) using **Multiple Linear Regression**, Python, Pandas, and scikit-learn.

## Overview

This project explores student academic data, preprocesses numerical and categorical variables, trains a regression model, and evaluates predictions on a held-out test set. It is a **final-grade prediction** model: `G1` and `G2` are used as inputs, so those grades must already be available when making a prediction. It is **not** an early-warning model.

## Dataset

Uses the **Student Performance** dataset from the UCI Machine Learning Repository, specifically the Portuguese-language dataset (`student-por.csv`). The dataset contains 649 student records and 33 original attributes.

- Source: https://archive.ics.uci.edu/dataset/320/student+performance
- Original paper: Cortez, P. and Silva, A. M. G. (2008), *Using Data Mining to Predict Secondary School Student Performance*.

Download the dataset from UCI and place `student-por.csv` in the `data/` folder. The CSV is not included in this starter package.

## Features and target

| Type | Columns |
| --- | --- |
| Numerical inputs | `G1`, `G2`, `studytime`, `absences`, `failures` |
| Categorical inputs | `school`, `sex` |
| Target | `G3` (final grade) |

## Methods

1. Data inspection and exploratory analysis (histogram, correlation matrix, scatter plot).
2. Train/test split (80/20, `random_state=42`).
3. One-hot encoding of categorical features, fitted **only on training data**.
4. Training with `sklearn.linear_model.LinearRegression`.
5. Evaluation using MAE, RMSE, R², actual-versus-predicted plot, and residual plot.
6. Prediction for a sample student.

## How to run

1. Install Python 3.10+ and Jupyter Notebook or use VS Code with the Jupyter extension.
2. Clone or download this repository.
3. Download `student-por.csv` from the dataset source and place it at `data/student-por.csv`.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Open `main.ipynb` and **Restart Kernel → Run All Cells** from the repository root.

## Results

Run the notebook to calculate the current MAE, RMSE, and R². **Results are intentionally not claimed here until the corrected notebook is freshly executed with the dataset.**

## Connection to gradient descent

The model learns a linear relationship of the form `ŷ = θ₀ + θ₁x₁ + ... + θₙxₙ`. A separate gradient descent exercise can help explain how coefficients can be optimized iteratively. **This notebook uses scikit-learn's least-squares solver, not a custom gradient descent training loop.**

## Limitations

- Earlier-period grades (`G1`, `G2`) must be available at prediction time.
- The dataset comes from Portuguese secondary schools; performance elsewhere may differ.
- Unconstrained linear regression can produce predictions outside the 0–20 grade range.
- The `G3 < 10` at-risk label in the notebook is a descriptive grouping, not a validated risk classifier.

## License and attribution

The project code is provided for educational demonstration. Cite the original UCI dataset and paper if reusing the data. Check the dataset's own terms before redistributing its files.
