# ============================================================
# Assignment — Week 04 · Day 21 (PM Session)
# PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
# Part C — Interview Ready (20%)
# ============================================================

import numpy as np


# ─────────────────────────────────────────
# Q1: Regression vs Classification — Written
# ─────────────────────────────────────────

print("=" * 60)
print("Q1: Regression vs Classification")
print("=" * 60)
print("""
Regression:
  Predicts a CONTINUOUS numeric output.
  The answer can be any real number within a range.

  Real-world examples:
    → Predicting house prices (₹42.5 lakhs)
    → Forecasting tomorrow's temperature (32.7°C)
    → Estimating a student's exam score (74.3/100)
    → Predicting monthly sales revenue

Classification:
  Predicts a DISCRETE class label from a fixed set.
  The answer is one of N predefined categories.

  Real-world examples:
    → Email spam detection (Spam / Not Spam)
    → Disease diagnosis (Diabetic / Not Diabetic)
    → Image recognition (Cat / Dog / Bird)
    → Loan approval (Approved / Rejected)

Key Difference:
  Regression  → "How much?" or "How many?" → continuous number
  Classification → "Which category?" → discrete label

  The decision is made by looking at the TARGET variable:
    - Continuous target  → Regression
    - Categorical target → Classification
""")


# ─────────────────────────────────────────
# Q2 (Coding): calculate_mse function
# ─────────────────────────────────────────

print("=" * 60)
print("Q2: calculate_mse(y_true, y_pred)")
print("=" * 60)


def calculate_mse(y_true, y_pred):
    """
    Compute Mean Squared Error between true and predicted values.

    Formula: MSE = (1/n) * Σ(y_true - y_pred)²

    Args:
        y_true : array of actual target values
        y_pred : array of predicted values

    Returns:
        MSE as a float
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    squared_errors = (y_true - y_pred) ** 2
    return np.mean(squared_errors)


# Test the function
y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 33, 37, 54])

mse_val  = calculate_mse(y_true, y_pred)
rmse_val = mse_val ** 0.5

print(f"\nActual    : {y_true}")
print(f"Predicted : {y_pred}")
print(f"\nMSE       : {mse_val:.4f}")
print(f"RMSE      : {rmse_val:.4f}  (sqrt of MSE, in original units)")

# Show individual squared errors
print(f"\nBreakdown:")
for yt, yp in zip(y_true, y_pred):
    err = (yt - yp) ** 2
    print(f"  ({yt} - {yp})² = {err}")
print(f"  Mean = {mse_val:.4f}")

# Perfect prediction
perfect_pred = y_true.copy()
print(f"\nPerfect prediction MSE : {calculate_mse(y_true, perfect_pred):.4f}  (expected: 0.0)")


# ─────────────────────────────────────────
# Q3: Bias–Variance Tradeoff — Written
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Q3: Bias–Variance Tradeoff")
print("=" * 60)
print("""
Bias–Variance Tradeoff:
  Every ML model's prediction error can be decomposed as:
  Total Error = Bias² + Variance + Irreducible Noise

Underfitting (High Bias, Low Variance):
  - The model is too simple to capture patterns in the data.
  - It makes strong, wrong assumptions about the data structure.
  - Performance is poor on BOTH training and test sets.
  - Signs: high training error, high test error, both similar.
  - Fix: increase model complexity, add more features, reduce
    regularisation strength.
  - Example: Using a straight line to fit a curved dataset.

Overfitting (Low Bias, High Variance):
  - The model is too complex and memorises the training data,
    including its noise.
  - It performs perfectly on training data but fails on new data.
  - Signs: very low training error but HIGH test error (large gap).
  - Fix: reduce complexity, add regularisation (L1/L2), get more
    training data, use dropout (for neural networks).
  - Example: A degree-9 polynomial fitting 20 noisy data points.

Optimal Model:
  - Sits at the "sweet spot" where test error is minimised.
  - Achieves low bias (captures real patterns) AND
    low variance (doesn't memorise noise).
  - Found via cross-validation, regularisation tuning, or
    learning curve analysis.

Summary:
  Simple model  → High bias,    Low variance  → Underfit
  Complex model → Low bias,     High variance → Overfit
  Optimal model → Balanced bias and variance  → Generalises well
""")
