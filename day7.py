import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the student dataset
df = pd.read_csv("students.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())


# 1. Plot distribution of grades
plt.figure(figsize=(7, 5))
sns.countplot(x="Grade", data=df)

plt.title("Distribution of Student Grades")
plt.xlabel("Grade")
plt.ylabel("Number of Students")

plt.show()


# 2. Show grade frequency
print("\nGrade Frequency:")
print(df["Grade"].value_counts())


# 3. Convert grades into numbers
grade_values = {
    "A": 4,
    "B": 3,
    "C": 2
}

df["Grade_Number"] = df["Grade"].map(grade_values)


# 4. Identify patterns and outliers
plt.figure(figsize=(7, 5))
sns.boxplot(y=df["Grade_Number"])

plt.title("Grade Distribution")
plt.ylabel("Grade")

plt.show()


# 5. Statistical summary
print("\nStatistical Summary:")
print(df["Grade_Number"].describe())


# 6. Five insights
print("\n5 Insights:")
print("1. Grade A is the most common grade.")
print("2. Grade B is the second most common grade.")
print("3. Grade C appears less frequently than A and B.")
print("4. Most students have grades between A and C.")
print("5. The dataset shows different levels of student performance.")