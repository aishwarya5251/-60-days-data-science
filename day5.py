# Day 5 - Dataset Analysis

# Import pandas library
import pandas as pd

# Load the dataset
df = pd.read_csv("student-mat.csv")

# Print the shape of the dataset
print("Shape of dataset:")
print(df.shape)

# Print the column names
print("\nColumns in dataset:")
print(df.columns)

# Print the first 5 rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Identify the target variable
target_variable = "G3"

print("\nTarget variable:")
print(target_variable)