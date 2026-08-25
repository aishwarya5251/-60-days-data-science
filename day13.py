"""
Day 13 - Overfitting & Regularization
Dataset: students.csv (Name, Grade) -- only 8 rows.

DESIGN CHOICE (explain this in your notebook/README):
With only 8 rows and 1 real feature, a plain Linear Regression can't
overfit in any visible way -- there's nothing complex enough to memorize.
To actually DEMONSTRATE overfitting, we deliberately expand name_length
into polynomial features (x, x^2, x^3, x^4, x^5) using only 6 training
points. This creates a classic "too many features, too little data"
setup -- exactly the scenario Ridge/Lasso regularization exist to fix.
This is an intentional teaching setup, not a real-world use case.

Steps:
1. Train a baseline model (plain Linear Regression on polynomial features)
2. Train Ridge and Lasso Regression on the same features
3. Compare train vs test performance
4. Identify signs of overfitting
5. Document model behavior differences
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------------------------------
# 1. LOAD & FEATURE-ENGINEER (same base logic as Day 10/11/12)
# -----------------------------------------------------------------
df = pd.read_csv("students.csv")

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df["grade_point"] = df["Grade"].map(grade_map)
df["name_length"] = df["Name"].str.len()

print("Full dataset:\n", df, "\n")

X = df[["name_length"]]
y = df["grade_point"]

# -----------------------------------------------------------------
# 2. TRAIN/TEST SPLIT (before expanding features, to avoid leakage)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"Train size: {len(X_train)}  Test size: {len(X_test)}\n")

# -----------------------------------------------------------------
# 3. EXPAND INTO POLYNOMIAL FEATURES (deliberately overcomplicate)
# -----------------------------------------------------------------
degree = 5
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale -- important for Ridge/Lasso, which are sensitive to feature scale,
# and essential here since x^5 values get huge compared to x.
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

print(f"Expanded {X_train.shape[1]} feature into {X_train_poly.shape[1]} "
      f"polynomial features (degree={degree}) using only {len(X_train)} "
      f"training rows -- a recipe for overfitting.\n")

# -----------------------------------------------------------------
# 4. TRAIN THREE MODELS ON THE SAME (OVERCOMPLICATED) FEATURES
# -----------------------------------------------------------------
models = {
    "Baseline (Linear Regression)": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
}

results = []

for name, model in models.items():
    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    # How large the coefficients got -- overfit models tend to blow these up
    coef_magnitude = np.abs(model.coef_).sum()

    results.append({
        "model": name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "sum_abs_coefficients": coef_magnitude,
    })

    print(f"--- {name} ---")
    print(f"  Train R^2: {train_r2:.3f}   Test R^2: {test_r2:.3f}")
    print(f"  Train MAE: {train_mae:.3f}  Test MAE: {test_mae:.3f}")
    print(f"  Sum |coefficients|: {coef_magnitude:.3f}")
    print()

# -----------------------------------------------------------------
# 5. COMPARISON TABLE
# -----------------------------------------------------------------
results_df = pd.DataFrame(results)
print("=== Train vs Test Performance Comparison ===")
print(results_df.to_string(index=False))

results_df.to_csv("day13_model_comparison.csv", index=False)
print("\nSaved: day13_model_comparison.csv")

# -----------------------------------------------------------------
# 6. IDENTIFY SIGNS OF OVERFITTING
# -----------------------------------------------------------------
print("\n=== Overfitting Diagnosis ===")
for row in results:
    gap = row["train_r2"] - row["test_r2"]
    verdict = "LIKELY OVERFITTING" if gap > 0.3 else "reasonable gap"
    print(f"{row['model']}: train-test R^2 gap = {gap:.3f} -> {verdict}")

print(
    "\nExpected pattern: the baseline Linear Regression (no regularization) "
    "should show a high train R^2 but poor/unstable test R^2, and large "
    "coefficient magnitudes (it's fitting noise in the 6 training points). "
    "Ridge should shrink coefficients and reduce the train-test gap. Lasso "
    "should shrink some coefficients to exactly zero, effectively dropping "
    "the least useful polynomial terms -- a form of automatic feature "
    "selection. Note: with n=8 total, exact numbers will vary run to run "
    "and shouldn't be over-interpreted -- the DIRECTION of the pattern "
    "(baseline overfits more than Ridge/Lasso) is the point of this exercise."
)