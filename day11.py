"""
Day 11 - Building Your First ML Pipeline
Dataset: students.csv (Name, Grade) -- only 8 rows.

IMPORTANT CAVEAT (keep this in your notebook/README write-up):
With only 8 rows, this pipeline demonstrates the MECHANICS of an ML
workflow (split -> train -> predict -> evaluate), not a genuinely
learnable pattern. Accuracy numbers here are not statistically
meaningful -- treat this as a "plumbing" exercise, and say so explicitly
in your reflection so it reads as intentional.

Steps:
1. Split into train/test sets
2. Select a baseline algorithm (Logistic Regression)
3. Train the model
4. Generate predictions on the test set
5. Analyze prediction quality
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------------------------------
# 1. LOAD & FEATURE-ENGINEER (same logic as Day 10)
# -----------------------------------------------------------------
df = pd.read_csv("students.csv")

grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df["grade_point"] = df["Grade"].map(grade_map)
df["name_length"] = df["Name"].str.len()
df["pass_flag"] = (df["grade_point"] >= 3).astype(int)  # B or better = pass, C = fail

print("Full dataset:\n", df, "\n")

# -----------------------------------------------------------------
# 2. DEFINE FEATURES (X) AND TARGET (y)
# -----------------------------------------------------------------
# We deliberately use name_length as the only predictor here, NOT
# grade_point -- using grade_point would be data leakage, since
# pass_flag is directly derived from it (the model would just be
# memorizing the mapping, not learning anything).
X = df[["name_length"]]
y = df["pass_flag"]

print("Features (X):\n", X.values.ravel())
print("Target (y):\n", y.values)
print(
    "\nNote: name_length has no real relationship to pass/fail -- "
    "this is intentional, it isolates the pipeline mechanics from "
    "any claim of a meaningful signal.\n"
)

# -----------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------------------------------------------
# With n=8, a typical 80/20 split leaves ~2 test rows. random_state
# fixed for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

# -----------------------------------------------------------------
# 4. SELECT & TRAIN A BASELINE MODEL
# -----------------------------------------------------------------
# Logistic Regression is a standard, interpretable baseline for
# binary classification problems.
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# 5. GENERATE PREDICTIONS
# -----------------------------------------------------------------
y_pred = model.predict(X_test)

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = y_pred
print("\nPredictions on test set:\n", results)

results.to_csv("day11_predictions.csv", index=False)
print("\nSaved: day11_predictions.csv")

# -----------------------------------------------------------------
# 6. ANALYZE PREDICTION QUALITY
# -----------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.2f}")
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

print(
    "\nReminder: with a 2-row test set, accuracy jumps in large "
    "increments (0%, 50%, 100%) and is NOT a reliable performance "
    "estimate. Report this honestly in your writeup as a limitation "
    "of the dataset size, not a reflection of model quality."
)