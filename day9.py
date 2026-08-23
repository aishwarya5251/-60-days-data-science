"""
Day 9 - Cleaning Messy Real-World Data
Dataset: students.csv (Name, Grade) -- originally only 8 clean rows.

DESIGN CHOICE (explain this in your notebook/README):
The original students.csv has no missing values, no duplicates, and
consistent formatting -- there's nothing to clean. To make Day 9's
task meaningful, this script first DELIBERATELY creates a messy
version (raw_students_messy.csv) with realistic real-world issues:
missing values, duplicate rows, and inconsistent text formatting
(casing, whitespace, mixed grade formats). It then walks through
identifying and fixing each issue, documenting every decision.

Steps:
1. Identify missing values
2. Handle null values with appropriate techniques
3. Detect duplicate records
4. Fix inconsistent data formats
5. Document every cleaning decision made
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------------
# 0. LOAD ORIGINAL CLEAN DATA
# -----------------------------------------------------------------
df_original = pd.read_csv("students.csv")
print("Original clean data:\n", df_original, "\n")

# -----------------------------------------------------------------
# 1. CREATE A DELIBERATELY MESSY VERSION (for demo purposes)
# -----------------------------------------------------------------
# We inject: a missing Name, a missing Grade, a duplicate row,
# inconsistent casing/whitespace in Name, and inconsistent Grade
# formatting (lowercase, trailing space, "A+" style).
messy_rows = [
    {"Name": "Aishwarya", "Grade": "A"},
    {"Name": "  rahul", "Grade": "b "},          # whitespace + lowercase
    {"Name": "Priya", "Grade": "A"},
    {"Name": "Ankit", "Grade": np.nan},           # missing grade
    {"Name": "Sneha", "Grade": "B"},
    {"Name": "Riya", "Grade": "a"},               # lowercase grade
    {"Name": "KARAN", "Grade": "C"},              # inconsistent casing
    {"Name": np.nan, "Grade": "B"},               # missing name
    {"Name": "Priya", "Grade": "A"},              # exact duplicate of row 2
    {"Name": "Aman", "Grade": "b"},               # lowercase grade
]
df_messy = pd.DataFrame(messy_rows)
df_messy.to_csv("raw_students_messy.csv", index=False)

print("=== MESSY (raw) data, as if freshly collected ===")
print(df_messy, "\n")

# -----------------------------------------------------------------
# 2. IDENTIFY MISSING VALUES
# -----------------------------------------------------------------
print("=== Step 1: Missing values ===")
missing_counts = df_messy.isnull().sum()
print(missing_counts, "\n")
print(f"Rows with any missing value:\n{df_messy[df_messy.isnull().any(axis=1)]}\n")

# -----------------------------------------------------------------
# 3. HANDLE NULL VALUES
# -----------------------------------------------------------------
df_clean = df_messy.copy()

# Decision: a missing Name means we can't identify the student at all --
# drop that row rather than guessing a name.
before = len(df_clean)
df_clean = df_clean.dropna(subset=["Name"])
print(f"Dropped {before - len(df_clean)} row(s) with missing Name "
      f"(no reasonable way to impute a person's identity).")

# Decision: a missing Grade is different -- we still know WHO the
# student is, so instead of dropping them we fill with a clearly
# labeled placeholder ("Unknown") rather than guessing a grade,
# which would fabricate a result that affects their record.
df_clean["Grade"] = df_clean["Grade"].fillna("Unknown")
print("Filled missing Grade values with 'Unknown' (labeled placeholder, "
      "not a guessed grade -- guessing a grade would be fabricating data).\n")

# -----------------------------------------------------------------
# 4. DETECT DUPLICATE RECORDS
# -----------------------------------------------------------------
print("=== Step 2: Duplicate records ===")
# Check exact duplicates first
exact_dupes = df_clean[df_clean.duplicated(keep=False)]
print(f"Exact duplicate rows:\n{exact_dupes}\n")

before = len(df_clean)
df_clean = df_clean.drop_duplicates(keep="first")
print(f"Removed {before - len(df_clean)} exact duplicate row(s), "
      f"kept the first occurrence.\n")

# -----------------------------------------------------------------
# 5. FIX INCONSISTENT DATA FORMATS
# -----------------------------------------------------------------
print("=== Step 3: Inconsistent formats ===")

# Name: strip whitespace, standardize to Title Case
print("Before Name cleanup:", df_clean["Name"].tolist())
df_clean["Name"] = df_clean["Name"].str.strip().str.title()
print("After Name cleanup: ", df_clean["Name"].tolist())
print("Decision: stripped leading/trailing whitespace and standardized "
      "casing to Title Case so 'KARAN', '  rahul', and 'Priya' are all "
      "consistently formatted.\n")

# Grade: strip whitespace, standardize to uppercase single letter
print("Before Grade cleanup:", df_clean["Grade"].tolist())
df_clean["Grade"] = df_clean["Grade"].str.strip().str.upper()
print("After Grade cleanup: ", df_clean["Grade"].tolist())
print("Decision: stripped whitespace and standardized casing so 'b ', "
      "'a', and 'B' all normalize to a consistent single-letter format.\n")

# Validate Grade values against an expected set
valid_grades = {"A", "B", "C", "D", "F", "UNKNOWN"}
invalid_grades = df_clean[~df_clean["Grade"].str.upper().isin(valid_grades)]
if len(invalid_grades) > 0:
    print(f"WARNING: unexpected grade values found:\n{invalid_grades}\n")
else:
    print("All Grade values fall within the expected set "
          f"{sorted(valid_grades)}.\n")

# -----------------------------------------------------------------
# 6. FINAL SUMMARY / DOCUMENTATION
# -----------------------------------------------------------------
print("=== Cleaning Summary ===")
print(f"Messy rows in:  {len(df_messy)}")
print(f"Clean rows out: {len(df_clean)}")
print("\nCleaning decisions applied:")
print("1. Dropped rows with missing Name (identity can't be imputed).")
print("2. Filled missing Grade with 'Unknown' label (avoids fabricating data).")
print("3. Removed exact duplicate rows, kept first occurrence.")
print("4. Standardized Name to stripped Title Case.")
print("5. Standardized Grade to stripped uppercase single letter.")
print("6. Validated all Grade values against expected set {A,B,C,D,F,Unknown}.")

print("\nFinal cleaned data:\n", df_clean.reset_index(drop=True))

df_clean.to_csv("students_cleaned_day9.csv", index=False)
print("\nSaved: students_cleaned_day9.csv")