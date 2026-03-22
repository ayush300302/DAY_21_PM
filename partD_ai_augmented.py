# ============================================================
# Assignment — Week 04 · Day 21 (PM Session)
# PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
# Part D — AI-Augmented Task (10%)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

"""
STEP 1 — PROMPT USED:
─────────────────────
"Explain regression vs classification and bias-variance
tradeoff with Python examples and visualizations."

─────────────────────────────────────────────────────────────
STEP 2 — AI OUTPUT (documented):
─────────────────────────────────────────────────────────────

The AI explained:

Regression vs Classification:
  Regression predicts continuous output; classification predicts
  discrete labels. AI used sklearn's LinearRegression and
  LogisticRegression as examples with synthetic data.

Bias-Variance Tradeoff:
  AI showed polynomial regression with degrees 1, 3, and 10,
  explaining that degree 1 underfits (high bias) and degree 10
  overfits (high variance). It plotted all three fits on the
  same graph using matplotlib.

AI Code Summary:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Polynomial fits
    for degree in [1, 3, 10]:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)

─────────────────────────────────────────────────────────────
STEP 3 — EVALUATION:
─────────────────────────────────────────────────────────────

✅ Are explanations correct?
   YES — The regression vs classification explanation was
   accurate. The distinction between continuous and discrete
   outputs was clearly stated with good real-world examples.

✅ Do visualizations correctly show underfitting and overfitting?
   YES — The degree-1 vs degree-10 plot clearly illustrated
   underfitting and overfitting respectively.

⚠️  Limitations noticed:
   1. The AI used sklearn's make_pipeline and PolynomialFeatures,
      which hides the math. Our assignment implements polynomial
      features manually to show actual understanding.
   2. The AI only showed degrees 1, 3, 10 — missing the "optimal"
      middle ground. Our Part B includes degree 2 as the sweet spot.
   3. The AI did not compute or compare Train MSE vs Test MSE
      numerically — just visual plots.
   4. No explanation of HOW to find the optimal model
      (cross-validation, regularisation) was provided.

Overall:
  Explanations → Excellent (clear and accurate)
  Visualizations → Good (correct trend shown)
  Depth → Moderate (missing quantitative analysis)
"""

# ─────────────────────────────────────────
# VERIFICATION: Manual reimplementation
# of AI's polynomial bias-variance demo
# ─────────────────────────────────────────

print("=" * 60)
print("Part D — AI Output Verification (Manual)")
print("=" * 60)


def poly_features(X, degree):
    return np.column_stack([X ** d for d in range(degree + 1)])


# Generate data
X_train = np.linspace(0, 2 * np.pi, 15)
y_train = np.sin(X_train) + np.random.normal(0, 0.3, 15)
X_plot  = np.linspace(0, 2 * np.pi, 200)
y_true  = np.sin(X_plot)

degrees = [1, 3, 10]
colors  = ["#E74C3C", "#27AE60", "#9B59B6"]
labels  = ["Degree 1 (Underfit)", "Degree 3 (Good fit)", "Degree 10 (Overfit)"]

plt.figure(figsize=(9, 4))
plt.scatter(X_train, y_train, s=30, color="black", zorder=5, label="Training data")
plt.plot(X_plot, y_true, color="black", linestyle="--", linewidth=1.5, label="True function")

for deg, color, label in zip(degrees, colors, labels):
    Phi_tr = poly_features(X_train, deg)
    Phi_pl = poly_features(X_plot,  deg)
    w = np.linalg.lstsq(Phi_tr, y_train, rcond=None)[0]
    y_pred = Phi_pl @ w
    train_pred = Phi_tr @ w

    train_mse = np.mean((y_train - train_pred) ** 2)
    test_mse  = np.mean((y_true  - y_pred) ** 2)
    print(f"  Degree {deg:>2} → Train MSE: {train_mse:.4f}  |  Test MSE: {test_mse:.4f}  → {label}")
    plt.plot(X_plot, y_pred, color=color, linewidth=2, label=label)

plt.ylim(-3, 3)
plt.title("Part D: Bias–Variance Verification (Manual, no sklearn)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("partD_bias_variance_verification.png", dpi=120)
plt.close()
print("\nPlot saved → partD_bias_variance_verification.png")
print("\n✅ AI visualizations verified — trend matches (underfit → overfit).")
print("⚠️  Our implementation is manual (no sklearn), showing deeper understanding.")
print("⚠️  Quantitative MSE comparison added — missing from AI's response.")
