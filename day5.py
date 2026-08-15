# Import pandas library
import pandas as pd

# Load the dataset
df = pd.read_csv("student_performance.csv")

# Display the shape of the dataset
# Shape tells us the number of rows and columns
print("Shape of Dataset:", df.shape)

# Display all column names
print("\nColumns in Dataset:")
print(df.columns)

# Display the first 5 rows of the dataset
print("\nFirst 5 Rows of Dataset:")
print(df.head())

# Identify the target variable
# For Student Performance dataset, G3 is the final grade
target_variable = "G3"

print("\nTarget Variable:", target_variable)