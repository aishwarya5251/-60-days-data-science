# Day 6 - Data Cleaning
# Goal:
# 1. Handle missing values
# 2. Remove duplicate rows
# 3. Fix data types
# 4. Create 2 new features
# 5. Save the cleaned dataset

import pandas as pd

# ---------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------

df = pd.read_csv("students.csv")

print("Original Dataset:")
print(df.head())

print("\nOriginal Shape:", df.shape)


# ---------------------------------------------------
# 2. Check missing values
# ---------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ---------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------

# Fill numerical missing values with the median
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values with the mode
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# ---------------------------------------------------
# 4. Remove duplicate rows
# ---------------------------------------------------

print("\nDuplicates before removal:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after removal:", df.duplicated().sum())


# ---------------------------------------------------
# 5. Fix data types
# ---------------------------------------------------

# Convert numerical columns to numeric type
# errors="coerce" converts invalid values into NaN

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ---------------------------------------------------
# 6. Create New Feature 1
# ---------------------------------------------------

# Total study-related score
# Change these column names if your dataset uses different names

if "G1" in df.columns and "G2" in df.columns:
    df["Average_G1_G2"] = (df["G1"] + df["G2"]) / 2


# ---------------------------------------------------
# 7. Create New Feature 2
# ---------------------------------------------------

# Total absences category

if "absences" in df.columns:
    df["Attendance_Status"] = df["absences"].apply(
        lambda x: "Good Attendance" if x <= 10 else "High Absence"
    )


# ---------------------------------------------------
# 8. Check the cleaned dataset
# ---------------------------------------------------

print("\nCleaned Dataset:")
print(df.head())

print("\nCleaned Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)


# ---------------------------------------------------
# 9. Save cleaned dataset
# ---------------------------------------------------

df.to_csv("cleaned_students.csv", index=False)

print("\nCleaned dataset saved successfully!")