# ============================================================
# Assignment — Week 04 · Day 21 (PM Session)
# PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
# Part B — Stretch Problem (30%)
# Bias–Variance Tradeoff
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ─────────────────────────────────────────
# Helper: Polynomial feature expansion
# ─────────────────────────────────────────

def poly_features(X, degree):
    """Create polynomial feature matrix up to given degree."""
    return np.column_stack([X ** d for d in range(0, degree + 1)])


def poly_fit_predict(X_train, y_train, X_test, degree):
    """Fit polynomial regression and return train + test predictions."""
    Phi_train = poly_features(X_train, degree)
    Phi_test  = poly_features(X_test, degree)
    w = np.linalg.lstsq(Phi_train, y_train, rcond=None)[0]
    return Phi_train @ w, Phi_test @ w


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ─────────────────────────────────────────
# B1: Simulate Bias–Variance Tradeoff
# Polynomial degree 1, 2, 5
# ─────────────────────────────────────────

print("=" * 60)
print("B1: Bias–Variance Tradeoff — Polynomial Degrees 1, 2, 5")
print("=" * 60)

# True function: y = sin(x) + noise
X_all  = np.linspace(0, 3 * np.pi, 200)
y_true = np.sin(X_all)

# Training data (fewer points + noise)
X_train = np.linspace(0, 3 * np.pi, 20)
y_train = np.sin(X_train) + np.random.normal(0, 0.3, size=20)

# Test data
X_test  = X_all
y_test  = y_true

degrees = [1, 2, 5, 9]
train_errors = []
test_errors  = []

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, deg in enumerate(degrees):
    y_train_pred, y_test_pred = poly_fit_predict(X_train, y_train, X_test, deg)

    tr_err = mse(y_train, y_train_pred)
    te_err = mse(y_test,  y_test_pred)
    train_errors.append(tr_err)
    test_errors.append(te_err)

    print(f"\nDegree {deg:>2}  →  Train MSE: {tr_err:.4f}  |  Test MSE: {te_err:.4f}", end="")
    if deg == 1:
        print("  ← Underfitting (high bias)")
    elif deg in [2]:
        print("  ← Good fit")
    elif deg == 5:
        print("  ← Starting to overfit")
    else:
        print("  ← Overfitting (high variance)")

    axes[i].scatter(X_train, y_train, s=25, color="#E74C3C", zorder=5, label="Train data")
    axes[i].plot(X_all, y_true, color="black", linestyle="--", linewidth=1.5, label="True function")
    axes[i].plot(X_test, y_test_pred, color="#4A90D9", linewidth=2, label=f"Degree {deg}")
    axes[i].set_title(f"Polynomial Degree {deg}\nTrain MSE={tr_err:.3f}  Test MSE={te_err:.3f}")
    axes[i].set_ylim(-3, 3)
    axes[i].legend(fontsize=8)

plt.suptitle("B1: Bias–Variance Tradeoff — Polynomial Degrees", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("b1_bias_variance_fits.png", dpi=120)
plt.close()
print("\nPlot saved → b1_bias_variance_fits.png")


# ─────────────────────────────────────────
# B2: Plot Training Error vs Model Complexity
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("B2: Training Error vs Model Complexity")
print("=" * 60)

all_degrees = list(range(1, 10))
tr_errs = []
te_errs = []

for deg in all_degrees:
    ytr, yte = poly_fit_predict(X_train, y_train, X_test, deg)
    tr_errs.append(mse(y_train, ytr))
    te_errs.append(mse(y_test,  yte))

plt.figure(figsize=(8, 4))
plt.plot(all_degrees, tr_errs, "o-", color="#27AE60", linewidth=2, label="Training Error")
plt.plot(all_degrees, te_errs, "s-", color="#E74C3C", linewidth=2, label="Test Error")
plt.axvline(2, color="gray", linestyle="--", linewidth=1.2, label="Optimal complexity")
plt.title("B2: Training vs Test Error by Polynomial Degree")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("MSE")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig("b2_error_vs_complexity.png", dpi=120)
plt.close()
print("Plot saved → b2_error_vs_complexity.png")

print(f"\nDegree  Train_MSE   Test_MSE")
print("-" * 35)
for d, tr, te in zip(all_degrees, tr_errs, te_errs):
    print(f"  {d:>2}    {tr:.5f}    {te:.5f}")


# ─────────────────────────────────────────
# B3: Explain Bias, Variance, Optimal Model
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("B3: Bias, Variance, Optimal Model — Explanation")
print("=" * 60)
print("""
What is Bias?
─────────────
  Bias is the error introduced by overly simplistic assumptions
  in the learning algorithm. A high-bias model cannot capture
  the true complexity of the data.

  Symptom  : High training error AND high test error.
  Cause    : Model is too simple (e.g., linear fit on curved data).
  Name     : UNDERFITTING
  Example  : Fitting a straight line to sinusoidal data.

What is Variance?
─────────────────
  Variance is the model's sensitivity to small fluctuations in
  the training data. A high-variance model fits training data
  extremely well but fails to generalise to new data.

  Symptom  : Very low training error but high test error.
  Cause    : Model is too complex (too many parameters).
  Name     : OVERFITTING
  Example  : A degree-9 polynomial memorising 20 training points.

Where does the Optimal Model lie?
───────────────────────────────────
  The optimal model minimises TOTAL ERROR = Bias² + Variance + Noise

  As complexity increases:
    → Bias decreases (model fits data better)
    → Variance increases (model becomes sensitive to noise)

  The sweet spot is where the test error is at its MINIMUM —
  low enough complexity to avoid overfitting, high enough to
  avoid underfitting.

  In our experiment: Degree 2 had the best test error balance.

  Strategies to find the sweet spot:
    1. Cross-validation (hold-out or k-fold)
    2. Regularisation (Ridge, Lasso) to penalise complexity
    3. Learning curves to diagnose bias vs variance
""")
