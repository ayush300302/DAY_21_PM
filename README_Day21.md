# 📘 Week 04 · Day 21 — Assignment Submission
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**


---

## 🔗 GitHub Repository Links

| Session | Repository Link |
|---------|----------------|
| **AM Session** — NumPy | https://github.com/ayush300302/DAY_21_AM |
| **PM Session** — Regression, Classification & Bias–Variance | https://github.com/ayush300302/DAY_21_PM |

---

## 🌅 AM Session — NumPy: Array Operations, Broadcasting, Vectorisation

---

### Part A — Concept Application

**Task 1 — 1D, 2D, 3D Arrays + Indexing:**
I created arrays of all three dimensions using `np.array()`. For indexing, I used standard `[row, col]` notation, extracted full rows and columns with slicing like `arr[:, 2]`, and pulled subarrays using `arr[0:2, 1:3]` on the 2D case.

**Task 2 — Basic Operations (No Loops):**
I used NumPy's element-wise operators (`+`, `-`, `*`, `/`) directly between two arrays. For statistics, I used `np.mean()`, `np.var()`, and `np.std()` — all vectorised with no manual iteration.

**Task 3 — Broadcasting:**
I demonstrated three cases: (1) adding a 1D row vector to a 2D matrix — each row gets the vector added; (2) multiplying a matrix by a scalar — broadcast to every element; (3) multiplying a matrix by a column vector of shape `(3,1)` — each row gets a different scalar. I explained the broadcasting rules: dimensions are matched right to left, and size-1 dimensions are virtually expanded.

**Task 4 — Vectorised Operations:**
I computed squares and cubes using `** 2` and `** 3` directly on arrays. Negative values were replaced using `np.where(arr < 0, 0, arr)`. Normalization was done using the min-max formula `(X - X.min()) / (X.max() - X.min())` — all without loops.

**Task 5 — Dataset Operations:**
I found the top 5 values by flattening and sorting in descending order. Row-wise and column-wise sums used `np.sum(axis=1)` and `np.sum(axis=0)`. Indices of values above threshold were found using `np.argwhere(dataset > threshold)`.

---

### Part B — Stretch Problem

**B1 — Matrix Operations:**
I used `np.dot(A, B)` for matrix multiplication, `A.T` for transpose, and `np.linalg.det(A)` for the determinant. Explained that a non-zero determinant means the matrix is invertible.

**B2 — Solving Linear Equations:**
I set up coefficient matrices and solved 2-variable and 3-variable systems using `np.linalg.solve()`. Verified each solution by computing `A @ x` and comparing with the constants vector using `np.allclose()`.

**B3 — Performance Comparison:**
I timed summation of 5 million elements using a Python `for` loop vs `np.sum()`. NumPy was significantly faster. I explained why: NumPy uses compiled C code, contiguous memory layout, SIMD CPU instructions, and avoids Python's per-iteration overhead.

---

### Part C — Interview Ready

**Q1 — Broadcasting:** Explained the three rules with shape examples and noted real-world use cases like subtracting mean per row and adding bias in neural networks.

**Q2 — normalize(X):** Implemented min-max normalization using `(X - X.min()) / (X.max() - X.min())`. Tested on both 1D and 2D arrays — confirmed min=0 and max=1 after transform.

**Q3 — Vectorisation vs Loops:** Explained that loops execute Python bytecode one step at a time while NumPy calls pre-compiled C/Fortran routines. Key reasons for speed: compiled C, contiguous memory, SIMD instructions, no type-checking overhead.

---

### Part D — AI-Augmented Task

**Prompt:** *"Explain NumPy broadcasting and vectorisation with practical Python examples."*

The AI gave correct examples for both broadcasting and vectorisation. I verified the code ran without errors. I additionally demonstrated column-vector broadcasting (shape `(2,1)`) and a broadcasting failure case — both of which the AI missed. Noted that `time.perf_counter()` is more accurate than `time.time()` for benchmarking.

---

## 🌆 PM Session — Regression, Classification & Bias–Variance Tradeoff

---

### Part A — Concept Application

**Task 1 & 2 — Synthetic Datasets + Model Training:**
I generated a regression dataset using `y = 3.5x + 7 + noise` and fitted a linear model using `np.linalg.lstsq()`. For classification, I used a threshold on a noisy signal to create binary labels and applied simple threshold logic. Both were plotted with matplotlib.

**Task 3 — Identify Problem Type:**
I identified four datasets (house prices, spam labels, temperatures, tumor types) and justified each as regression or classification based on whether the target variable is continuous or discrete.

**Task 4 — Manual Linear Regression + MSE:**
I implemented least squares manually: computed slope `w` using the covariance formula and intercept `b = ȳ - w·x̄`. Predicted outputs with `y = wx + b` and computed MSE as `mean((y_true - y_pred)²)`.

**Task 5 — Manual Classification + Accuracy:**
I applied a threshold of 60 on a score dataset and classified each sample as 0 or 1. Accuracy was computed by counting correct predictions manually and dividing by total samples.

**Task 6 — Comparison Table:**
I compared regression and classification across output type, use cases, loss functions, and evaluation metrics in a formatted table.

---

### Part B — Stretch Problem

**B1 — Bias–Variance Simulation:**
I fitted polynomial models of degrees 1, 2, 5, and 9 on a `sin(x)` dataset with noise. Implemented `poly_features()` manually using `np.column_stack`. Observed: degree 1 underfits (high bias), degrees 2–3 are optimal, degree 9 overfits (high variance).

**B2 — Error vs Complexity Plot:**
I plotted train and test MSE for degrees 1–9. The training error decreases monotonically while test error forms a U-shape — confirming the bias-variance tradeoff. Optimal point was at degree 2.

**B3 — Bias, Variance, Optimal Model:**
Explained bias as error from overly simple assumptions (underfit), variance as sensitivity to training data noise (overfit), and the optimal model as the sweet spot minimising total error = Bias² + Variance + Noise.

---

### Part C — Interview Ready

**Q1 — Regression vs Classification:** Explained with real-world examples — regression for price/temperature prediction, classification for spam/disease detection. Key distinction: continuous vs discrete output.

**Q2 — calculate_mse():** Implemented as `mean((y_true - y_pred)²)` using NumPy. Tested with known values and verified zero MSE for perfect predictions.

**Q3 — Bias–Variance Tradeoff:** Explained underfitting (high bias — model too simple, poor on train and test), overfitting (high variance — memorises noise, fails on test), and how to find the optimal model via cross-validation and regularisation.

---

### Part D — AI-Augmented Task

**Prompt:** *"Explain regression vs classification and bias-variance tradeoff with Python examples and visualizations."*

The AI gave accurate explanations and correct sklearn-based visualizations. I verified the bias-variance trend was correctly shown. I noted limitations: sklearn's `make_pipeline` hides the math, only 3 degrees were shown (missing the optimal middle ground), and no quantitative MSE comparison was made. I reimplemented everything manually to demonstrate deeper understanding.

---

## 📁 Repository Structure

```
├── AM/
│   ├── partA_numpy_operations.py      # Arrays, indexing, broadcasting, vectorisation
│   ├── partB_matrix_performance.py    # Matrix ops, linear equations, speed comparison
│   ├── partC_interview_ready.py       # Broadcasting Q&A, normalize(), vectorisation
│   └── partD_ai_augmented.py         # AI prompt verified, extra cases demonstrated
│
├── PM/
│   ├── partA_regression_classification.py  # Datasets, models, MSE, accuracy, comparison
│   ├── partB_bias_variance.py              # Polynomial fits, error plot, explanations
│   ├── partC_interview_ready.py            # MSE function, regression vs classification
│   └── partD_ai_augmented.py             # AI output verified, manual reimplementation
│
└── README.md
```

---

*Submitted by: Ayush Patil | IIT Gandhinagar*
