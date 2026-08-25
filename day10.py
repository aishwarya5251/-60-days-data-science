"""
Day 10 - Feature Engineering
Dataset: students.csv  ->  columns: Name (str), Grade (str: A/B/C/D/F)

Steps:
1. Identify categorical & numerical features
2. Encode categorical data
3. Scale numerical features
4. Create 2+ derived features
5. Compare model readiness before vs after
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------
df = pd.read_csv("students.csv")

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nData:\n", df)
print("\nMissing values:\n", df.isnull().sum())

df_raw = df.copy()  # untouched copy for the "before" comparison

# -----------------------------------------------------------------
# 2. IDENTIFY CATEGORICAL VS NUMERICAL FEATURES
# -----------------------------------------------------------------
# Name  -> identifier, not a predictive feature on its own (unique per row)
# Grade -> categorical, and ORDINAL (F < D < C < B < A), so it should be
#          mapped to numbers rather than one-hot encoded.
# There are no raw numerical columns in this dataset -- we'll create some
# in step 4 (derived features).

identifier_cols = ["Name"]
categorical_cols = ["Grade"]
numerical_cols = []  # none originally

print("\nIdentifier columns:", identifier_cols)
print("Categorical columns:", categorical_cols)
print("Numerical columns (raw):", numerical_cols)

# -----------------------------------------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# -----------------------------------------------------------------
# Grade is ordinal -> ordinal/label encode it into a meaningful numeric
# scale (GPA-style points), rather than one-hot encoding, which would
# throw away the natural A > B > C > D > F ordering.
grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df["grade_point"] = df["Grade"].map(grade_map)

# For comparison, here's what one-hot encoding Grade would look like
# (kept as a separate demo column set, not used for the ordinal analysis):
grade_onehot = pd.get_dummies(df["Grade"], prefix="Grade")

# -----------------------------------------------------------------
# 4. CREATE DERIVED FEATURES (at least 2)
# -----------------------------------------------------------------
# Derived feature 1: grade_point (numeric encoding of ordinal Grade) -- done above

# Derived feature 2: name_length -- a numeric feature derived from Name
df["name_length"] = df["Name"].str.len()

# Derived feature 3: pass_flag -- binary label from grade_point
df["pass_flag"] = (df["grade_point"] >= 2).astype(int)  # C or better = pass

numerical_cols = ["grade_point", "name_length"]  # now we have numeric features to scale

# -----------------------------------------------------------------
# 5. FEATURE SCALING (numerical columns)
# -----------------------------------------------------------------
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

# -----------------------------------------------------------------
# 6. BUILD FINAL MODEL-READY DATASET
# -----------------------------------------------------------------
df_encoded = pd.concat(
    [df_scaled[["Name", "grade_point", "name_length", "pass_flag"]], grade_onehot],
    axis=1,
)

# -----------------------------------------------------------------
# 7. BEFORE vs AFTER COMPARISON
# -----------------------------------------------------------------
print("\n--- BEFORE Feature Engineering ---")
print(df_raw)
print("Shape:", df_raw.shape)
print("Numeric columns usable by an ML model: 0")

print("\n--- AFTER Feature Engineering ---")
print(df_encoded)
print("Shape:", df_encoded.shape)
print("Numeric columns usable by an ML model:", len(numerical_cols) + 1 + len(grade_onehot.columns))
print("New columns added:", set(df_encoded.columns) - set(df_raw.columns))

# -----------------------------------------------------------------
# 8. SAVE FEATURE-ENGINEERED DATASET
# -----------------------------------------------------------------
df_encoded.to_csv("students_feature_engineered.csv", index=False)
print("\nSaved: students_feature_engineered.csv")