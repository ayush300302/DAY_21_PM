# ============================================================
# Assignment — Week 04 · Day 21 (PM Session)
# PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
# Part A — Concept Application (40%)
# Topic: Regression, Classification, Bias–Variance Tradeoff
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ─────────────────────────────────────────
# Q1 & Q2: Synthetic Datasets
# Regression (continuous) + Classification (binary)
# ─────────────────────────────────────────

print("=" * 60)
print("Q1 & Q2: Synthetic Datasets — Regression & Classification")
print("=" * 60)

# ── Regression Dataset ──────────────────
X_reg = np.linspace(0, 10, 100)
noise = np.random.normal(0, 1.5, size=100)
y_reg = 3.5 * X_reg + 7 + noise          # true: y = 3.5x + 7 + noise

# Simple Linear Regression (manual — no sklearn)
# Least squares: w = (X^T X)^-1 X^T y
X_b = np.column_stack([np.ones(len(X_reg)), X_reg])   # add bias column
w = np.linalg.lstsq(X_b, y_reg, rcond=None)[0]
y_pred_reg = X_b @ w

mse_reg = np.mean((y_reg - y_pred_reg) ** 2)
print(f"\nRegression:")
print(f"  Learned equation : y = {w[1]:.4f}x + {w[0]:.4f}")
print(f"  True equation    : y = 3.5x + 7")
print(f"  MSE              : {mse_reg:.4f}")

# Plot regression
plt.figure(figsize=(7, 4))
plt.scatter(X_reg, y_reg, s=15, color="#4A90D9", alpha=0.6, label="Data points")
plt.plot(X_reg, y_pred_reg, color="red", linewidth=2, label=f"Fit: y={w[1]:.2f}x+{w[0]:.2f}")
plt.title("Q2: Linear Regression on Synthetic Data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("q2_regression.png", dpi=120)
plt.close()
print("  Plot saved → q2_regression.png")

# ── Classification Dataset ──────────────
X_cls = np.random.randn(200)
y_cls = (X_cls + np.random.normal(0, 0.5, 200) > 0).astype(int)  # binary: 0 or 1

# Simple threshold classification
threshold = 0.0
y_pred_cls = (X_cls > threshold).astype(int)
accuracy = np.mean(y_pred_cls == y_cls) * 100

print(f"\nClassification:")
print(f"  Threshold used   : {threshold}")
print(f"  Accuracy         : {accuracy:.2f}%")

# Plot classification
plt.figure(figsize=(7, 4))
colors = np.where(y_cls == 1, "#27AE60", "#E74C3C")
plt.scatter(range(len(X_cls)), X_cls, c=colors, s=15, alpha=0.7)
plt.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
plt.title("Q2: Binary Classification on Synthetic Data")
plt.xlabel("Sample Index")
plt.ylabel("Feature Value")
plt.legend()
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#27AE60", label="Class 1"),
                   Patch(facecolor="#E74C3C", label="Class 0")]
plt.legend(handles=legend_elements + [plt.Line2D([0],[0], color='black',
           linestyle='--', label=f'Threshold={threshold}')])
plt.tight_layout()
plt.savefig("q2_classification.png", dpi=120)
plt.close()
print("  Plot saved → q2_classification.png")


# ─────────────────────────────────────────
# Q3: Identify regression vs classification
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Q3: Identify Problem Type from Target Variable")
print("=" * 60)

datasets = {
    "House prices (₹ in lakhs)": [45.2, 62.8, 38.5, 90.1, 55.3],
    "Email spam (0=no, 1=yes)" : [0, 1, 0, 0, 1],
    "Temperature tomorrow (°C)": [28.3, 31.0, 25.7, 29.5, 33.2],
    "Tumor type (0=benign,1=malignant)": [0, 1, 1, 0, 1],
}

for name, targets in datasets.items():
    unique = set(targets)
    if unique == {0, 1} or len(unique) <= 3:
        kind = "CLASSIFICATION"
        reason = "target has discrete class labels"
    else:
        kind = "REGRESSION"
        reason = "target is a continuous numeric value"
    print(f"\n  Dataset : {name}")
    print(f"  Type    : {kind}")
    print(f"  Reason  : {reason}")


# ─────────────────────────────────────────
# Q4: Manual Linear Regression
# y = wx + b, compute MSE
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Q4: Manual Linear Regression + MSE")
print("=" * 60)

# Simple dataset: study hours vs exam score
hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
scores = np.array([52, 55, 60, 65, 68, 72, 78, 82, 85, 92], dtype=float)

# Manual least squares
n     = len(hours)
x_bar = np.mean(hours)
y_bar = np.mean(scores)

w = np.sum((hours - x_bar) * (scores - y_bar)) / np.sum((hours - x_bar) ** 2)
b = y_bar - w * x_bar

y_pred_manual = w * hours + b
mse_manual    = np.mean((scores - y_pred_manual) ** 2)

print(f"\nHours  : {hours}")
print(f"Scores : {scores}")
print(f"\nLearned: y = {w:.4f}x + {b:.4f}")
print(f"MSE    : {mse_manual:.4f}")

for h, s, p in zip(hours, scores, y_pred_manual):
    print(f"  Hours={h:.0f}  Actual={s:.0f}  Predicted={p:.2f}  Error={abs(s-p):.2f}")


# ─────────────────────────────────────────
# Q5: Manual Classification with threshold
# Compute accuracy manually
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Q5: Manual Classification + Accuracy")
print("=" * 60)

scores_data = np.array([45, 72, 38, 91, 55, 60, 48, 83, 67, 30,
                        88, 52, 74, 41, 95, 63, 50, 77, 35, 86])
true_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
                        1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

threshold = 60
predicted = (scores_data >= threshold).astype(int)

correct = 0
for pred, true in zip(predicted, true_labels):
    if pred == true:
        correct += 1
accuracy_manual = correct / len(true_labels) * 100

print(f"\nScores     : {scores_data}")
print(f"True Labels: {true_labels}")
print(f"Threshold  : {threshold}")
print(f"Predicted  : {predicted}")
print(f"Correct    : {correct}/{len(true_labels)}")
print(f"Accuracy   : {accuracy_manual:.2f}%")


# ─────────────────────────────────────────
# Q6: Regression vs Classification — Compare
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Q6: Regression vs Classification — Comparison")
print("=" * 60)
print("""
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│ Aspect              │ Regression               │ Classification           │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Output Type         │ Continuous value         │ Discrete class label     │
│ Example Output      │ ₹52,300 / 37.5°C         │ Spam/Not Spam, 0/1       │
│ Use Cases           │ Price, temp, sales       │ Fraud detect, diagnosis  │
│ Evaluation Metric   │ MSE, RMSE, MAE, R²       │ Accuracy, F1, AUC-ROC    │
│ Loss Function       │ Mean Squared Error       │ Cross-Entropy Loss       │
│ Common Models       │ Linear, Polynomial Reg.  │ Logistic Reg., SVM, Tree │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
""")
