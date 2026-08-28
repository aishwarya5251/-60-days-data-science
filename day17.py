# Day 17 - Loan Approval Prediction with Decision Trees
# IMPORT LIBRARIES
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 1. LOAD OR CREATE DATASET
print("\nLoading dataset...")

if not os.path.exists("loan_data.csv") or os.path.getsize("loan_data.csv") == 0:

    print("loan_data.csv is missing or empty.")
    print("Creating sample loan dataset...")

    data = {
        "Age": [
            25, 32, 40, 28, 45, 23, 36, 50, 29, 31,
            42, 27, 38, 48, 26, 35, 44, 30, 39, 24,
            41, 33, 37, 52, 28, 34, 46, 22, 43, 31
        ],

        "Income": [
            45000, 60000, 80000, 35000, 90000, 28000,
            70000, 100000, 40000, 55000, 65000, 30000,
            75000, 85000, 32000, 58000, 62000, 50000,
            72000, 27000, 95000, 48000, 68000, 110000,
            36000, 53000, 88000, 25000, 78000, 46000
        ],

        "CreditScore": [
            650, 720, 750, 580, 780, 550, 710, 800,
            600, 680, 690, 570, 740, 770, 590, 700,
            660, 640, 730, 540, 790, 620, 710, 820,
            560, 670, 760, 520, 740, 610
        ],

        "LoanAmount": [
            15000, 20000, 30000, 18000, 25000, 12000,
            22000, 40000, 25000, 18000, 35000, 20000,
            28000, 50000, 22000, 20000, 40000, 18000,
            30000, 15000, 35000, 25000, 32000, 45000,
            30000, 20000, 55000, 10000, 35000, 28000
        ],

        "EmploymentYears": [
            2, 5, 10, 1, 15, 1, 8, 20, 3, 4,
            12, 2, 9, 18, 1, 7, 14, 5, 11, 1,
            13, 6, 8, 25, 2, 6, 16, 0, 14, 4
        ],

        "LoanApproved": [
            1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
            1, 0, 1, 1, 0, 1, 0, 1, 1, 0,
            1, 0, 1, 1, 0, 1, 1, 0, 1, 0
        ]
    }

    df = pd.DataFrame(data)

    df.to_csv("loan_data.csv", index=False)

    print("loan_data.csv created successfully!")

else:

    df = pd.read_csv("loan_data.csv")

    print("loan_data.csv loaded successfully!")
# DATASET INFORMATION
print("\nDataset Preview:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Statistics:")
print(df.describe())
# 2. DEFINE FEATURES AND TARGET
X = df.drop("LoanApproved", axis=1)
y = df["LoanApproved"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("LoanApproved")
# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
# 4. TRAIN DECISION TREE
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")
# 5. MAKE PREDICTIONS
y_pred = model.predict(X_test)

# 6. MODEL EVALUATION
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# 7. SAVE PREDICTIONS
results = X_test.copy()

results["Actual_LoanApproval"] = y_test.values
results["Predicted_LoanApproval"] = y_pred

results.to_csv(
    "day17_predictions.csv",
    index=False
)

print("\nPredictions saved to day17_predictions.csv")
# 8. VISUALIZE DECISION TREE
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Rejected", "Approved"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Loan Approval Decision Tree")

plt.savefig(
    "day17_decision_tree.png",
    bbox_inches="tight"
)

plt.show()

print("\nDecision tree visualization saved!")
# 9. FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)
# 10. FEATURE IMPORTANCE VISUALIZATION
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Loan Approval Prediction")

plt.gca().invert_yaxis()

plt.savefig(
    "day17_feature_importance.png",
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance visualization saved!")
# 11. CHECK FOR OVERFITTING
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print("\nOverfitting Analysis")

print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Testing Accuracy: {test_accuracy:.2%}")

difference = train_accuracy - test_accuracy

print(f"Accuracy Difference: {difference:.2%}")

if difference > 0.15:

    overfitting_result = (
        "Possible overfitting detected. "
        "The training accuracy is significantly higher "
        "than the testing accuracy."
    )

else:

    overfitting_result = (
        "No major overfitting detected. "
        "Training and testing accuracy are relatively close."
    )

print("\n" + overfitting_result)
# 12. SAVE RESULTS

with open("day17_results.txt", "w") as file:

    file.write("DAY 17 - LOAN APPROVAL PREDICTION\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Model Accuracy: {accuracy:.2%}\n")
    file.write(f"Training Accuracy: {train_accuracy:.2%}\n")
    file.write(f"Testing Accuracy: {test_accuracy:.2%}\n\n")

    file.write("FEATURE IMPORTANCE\n")
    file.write("-" * 30 + "\n")

    for _, row in feature_importance.iterrows():

        file.write(
            f"{row['Feature']}: "
            f"{row['Importance']:.4f}\n"
        )

    file.write("\nOVERFITTING ANALYSIS\n")
    file.write("-" * 30 + "\n")

    file.write(overfitting_result)


print("\nResults saved to day17_results.txt")
# 13. SAMPLE PREDICTION
sample_applicant = pd.DataFrame({
    "Age": [30],
    "Income": [55000],
    "CreditScore": [700],
    "LoanAmount": [20000],
    "EmploymentYears": [5]
})

prediction = model.predict(sample_applicant)

print("\nSample Applicant Prediction:")

if prediction[0] == 1:

    print("Loan Prediction: APPROVED")

else:

    print("Loan Prediction: REJECTED")


print("\nDay 17 Completed Successfully!")