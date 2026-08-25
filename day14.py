"""
Day 14 - Sprint Review & Real-World Thinking
Remove the most important feature, retrain, and compare performance.

CONTEXT (read before running):
This dataset (students.csv: Name, Grade -- 8 rows) has exactly ONE real
feature: name_length, engineered from the Name column. Day 13 artificially
expanded that single feature into 5 polynomial terms (x, x^2, x^3, x^4, x^5)
to demonstrate overfitting with Ridge/Lasso regularization.

That means there ISN'T a normal "pick the top feature out of several"
situation here. name_length IS the only real signal the model has --
everything else is a mathematical expansion of it. So "removing the most
important feature" in this project means removing the model's ONLY source
of information, not just weakening it a bit.

This is intentional and worth keeping, not fixing: it's a genuinely useful
real-world lesson (see Sprint 2 reflection printed at the end) about what
happens when a pipeline loses its one meaningful signal.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------------------------------
# 1. LOAD & FEATURE-ENGINEER (identical to Day 13)
# -----------------------------------------------------------------
df = pd.read_csv("students.csv")

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df["grade_point"] = df["Grade"].map(grade_map)
df["name_length"] = df["Name"].str.len()

print("Full dataset:\n", df, "\n")

X = df[["name_length"]]
y = df["grade_point"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

degree = 5
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

# -----------------------------------------------------------------
# 2. TRAIN ALL THREE CANDIDATES (same as Day 13) AND PICK THE BEST
# -----------------------------------------------------------------
candidates = {
    "Baseline (Linear Regression)": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
}

candidate_results = []
fitted_models = {}

for name, model in candidates.items():
    model.fit(X_train_poly, y_train)
    fitted_models[name] = model
    test_pred = model.predict(X_test_poly)
    test_r2 = r2_score(y_test, test_pred)
    train_pred = model.predict(X_train_poly)
    train_r2 = r2_score(y_train, train_pred)
    candidate_results.append({
        "model": name,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "overfit_gap": train_r2 - test_r2,
    })

candidates_df = pd.DataFrame(candidate_results).sort_values(
    "test_r2", ascending=False
)
print("=== Day 13 candidates (re-evaluated) ===")
print(candidates_df.to_string(index=False))
print()

best_name = candidates_df.iloc[0]["model"]
best_model = fitted_models[best_name]
old_r2 = candidates_df.iloc[0]["test_r2"]
old_predictions = best_model.predict(X_test_poly)
old_mae = mean_absolute_error(y_test, old_predictions)

print(f"Best-performing model: {best_name}")
print(f"Old R^2 (test): {old_r2:.4f}")
print(f"Old MAE (test): {old_mae:.4f}\n")

# -----------------------------------------------------------------
# 3. FEATURE IMPORTANCE
# -----------------------------------------------------------------
# All 5 polynomial columns are mathematically derived from name_length,
# so there is no independent second feature to rank against it.
poly_feature_names = poly.get_feature_names_out(["name_length"])
importance = pd.Series(
    np.abs(best_model.coef_), index=poly_feature_names
).sort_values(ascending=False)

print("=== Coefficient magnitude by polynomial term ===")
print(importance)
print(
    "\nEvery one of these terms comes from a single raw feature: "
    "name_length. So the 'most important feature' in this project "
    "is name_length itself -- there is nothing else to fall back on.\n"
)

important_feature = "name_length"
print("Removing feature:", important_feature)
print()

# -----------------------------------------------------------------
# 4. REMOVE THE ONLY REAL FEATURE AND TRY TO RETRAIN
# -----------------------------------------------------------------
X_reduced = df.drop(columns=["name_length", "Name", "Grade", "grade_point"])
print(f"Columns remaining after removal: {list(X_reduced.columns)}")
print(f"X_reduced has {X_reduced.shape[1]} usable feature column(s).\n")

if X_reduced.shape[1] == 0:
    print(
        "No features remain. A model literally cannot be trained on zero "
        "columns (sklearn raises a ValueError if you try). This is the "
        "real-world failure mode: the pipeline must FALL BACK to a naive "
        "baseline -- here, a DummyRegressor predicting the mean grade_point "
        "for every student, since there is no longer any input to learn from.\n"
    )
    _, _, y_train_new, y_test_new = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    new_model = DummyRegressor(strategy="mean")
    new_model.fit(np.empty((len(y_train_new), 0)), y_train_new)
    new_predictions = new_model.predict(np.empty((len(y_test_new), 0)))
else:
    # (kept for reuse if this script is adapted to a dataset with
    # more than one real feature later)
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_reduced, y, test_size=0.25, random_state=42
    )
    new_model = LinearRegression()
    new_model.fit(X_train_new, y_train_new)
    new_predictions = new_model.predict(X_test_new)

new_r2 = r2_score(y_test_new, new_predictions)
new_mae = mean_absolute_error(y_test_new, new_predictions)

print("=== AFTER removing 'name_length' (fallback model) ===")
print("New R^2 (test):", round(new_r2, 4))
print("New MAE (test):", round(new_mae, 4))
print()

# -----------------------------------------------------------------
# 5. COMPARISON
# -----------------------------------------------------------------
r2_change = new_r2 - old_r2
mae_change = new_mae - old_mae

print("=== PERFORMANCE COMPARISON ===")
print(f"R^2 change:  {r2_change:+.4f}")
print(f"MAE change:  {mae_change:+.4f}")

comparison_df = pd.DataFrame({
    "Metric": ["R2 Score (test)", "MAE (test)"],
    "Before (with name_length)": [old_r2, old_mae],
    "After (name_length removed, fallback to mean)": [new_r2, new_mae],
})
print()
print(comparison_df.to_string(index=False))
comparison_df.to_csv("day14_performance_comparison.csv", index=False)
print("\nSaved: day14_performance_comparison.csv")

# Day 14 - Sprint Review & Real-World 