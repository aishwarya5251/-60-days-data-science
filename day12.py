"""
Day 12 - Regression Modeling
Dataset: students.csv (Name, Grade) -- only 8 rows.

IMPORTANT CAVEAT (keep this in your notebook/README write-up):
Linear Regression wants a genuinely continuous target. The closest thing
in this dataset is grade_point, which only takes 3 distinct values
(2, 3, 4) across 8 rows -- so this is a mechanics demo (how to train,
visualize, and interpret a regression model), not a claim that
name_length actually predicts grade. Say this explicitly in your writeup.

Steps:
1. Train a Linear Regression model
2. Understand input (X) vs output (y) relationship
3. Visualize the prediction line and errors (residuals)
4. Interpret model coefficients
5. Analyze prediction accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------
# 1. LOAD & FEATURE-ENGINEER (same logic as Day 10/11)
# -----------------------------------------------------------------
df = pd.read_csv("students.csv")

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df["grade_point"] = df["Grade"].map(grade_map)
df["name_length"] = df["Name"].str.len()

print("Full dataset:\n", df, "\n")

# -----------------------------------------------------------------
# 2. DEFINE INPUT (X) AND OUTPUT (y)
# -----------------------------------------------------------------
# X: name_length  (input feature)
# y: grade_point  (continuous-ish target we're regressing on)
X = df[["name_length"]]
y = df["grade_point"]

print("Input (X) -> name_length:", X.values.ravel())
print("Output (y) -> grade_point:", y.values)
print(
    "\nNote: there is no real causal relationship between a student's "
    "name length and their grade. Any pattern the model finds here is "
    "coincidental, given n=8. This is a deliberate choice to keep the "
    "demo honest -- see the caveat at the top of this file.\n"
)

# -----------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

# -----------------------------------------------------------------
# 4. TRAIN THE LINEAR REGRESSION MODEL
# -----------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# 5. INTERPRET MODEL COEFFICIENTS
# -----------------------------------------------------------------
slope = model.coef_[0]
intercept = model.intercept_

print(f"\nModel equation: grade_point = {slope:.3f} * name_length + {intercept:.3f}")
print(
    f"Interpretation: for every 1-letter increase in name length, predicted "
    f"grade_point changes by {slope:.3f} points. Given the coincidental "
    f"nature of this relationship (see caveat), this coefficient should "
    f"NOT be read as a meaningful real-world effect."
)

# -----------------------------------------------------------------
# 6. GENERATE PREDICTIONS ON TEST SET
# -----------------------------------------------------------------
y_pred_test = model.predict(X_test)

results = X_test.copy()
results["actual_grade_point"] = y_test.values
results["predicted_grade_point"] = y_pred_test
print("\nPredictions on test set:\n", results)

results.to_csv("day12_predictions.csv", index=False)
print("\nSaved: day12_predictions.csv")

# -----------------------------------------------------------------
# 7. ANALYZE PREDICTION ACCURACY
# -----------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"\nMAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R^2: {r2:.3f}")
print(
    "\nWith only 2 test points, R^2 is not statistically reliable -- "
    "it can even go negative if the line happens to fit worse than a "
    "flat average. Report this as a dataset-size limitation."
)

# -----------------------------------------------------------------
# 8. VISUALIZE PREDICTION LINE + ERRORS
# -----------------------------------------------------------------
# Fit a line across the full range of X for a smooth plot
x_range = np.linspace(X["name_length"].min() - 1, X["name_length"].max() + 1, 100).reshape(-1, 1)
x_range_df = pd.DataFrame(x_range, columns=["name_length"])
y_line = model.predict(x_range_df)

plt.figure(figsize=(8, 6))

# All data points (train = blue, test = orange)
plt.scatter(X_train, y_train, color="steelblue", label="Train data", s=80, zorder=3)
plt.scatter(X_test, y_test, color="darkorange", label="Test data (actual)", s=80, zorder=3)

# Regression line
plt.plot(x_range, y_line, color="black", linestyle="--", label="Regression line", zorder=2)

# Predictions on test set + error lines (residuals)
plt.scatter(X_test, y_pred_test, color="red", marker="x", s=100, label="Test predictions", zorder=4)
for xi, y_actual, y_hat in zip(X_test["name_length"], y_test, y_pred_test):
    plt.plot([xi, xi], [y_actual, y_hat], color="red", linestyle=":", linewidth=1.5, zorder=1)

plt.xlabel("name_length (input)")
plt.ylabel("grade_point (output)")
plt.title("Linear Regression: name_length -> grade_point\n(red dotted lines = prediction errors)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("day12_regression_plot.png", dpi=150)
print("\nSaved: day12_regression_plot.png")
plt.show()