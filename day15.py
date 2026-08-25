"""
Day 15 - Predicting Student Pass/Fail with Logistic Regression

Classification Foundations

This project converts the existing students.csv Grade column
into a binary classification target.

A/B -> Pass (1)
C/D/F -> Fail (0)

Feature:
    name_length

Model:
    Logistic Regression

Outputs:
    day15_predictions.csv
    day15_confusion_matrix.png
    day15_results.txt
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("students.csv")

print("=" * 60)
print("DAY 15 - CLASSIFICATION")
print("=" * 60)

print("\nOriginal dataset:")
print(df)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Convert Grade into binary target
# A and B = Pass
# C, D and F = Fail

df["Pass"] = df["Grade"].map({
    "A": 1,
    "B": 1,
    "C": 0,
    "D": 0,
    "F": 0
})

# Use name length as the feature
df["name_length"] = df["Name"].str.len()

print("\nDataset after feature engineering:")
print(df)


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

X = df[["name_length"]]
y = df["Pass"]

print("\nFeature:")
print("name_length")

print("\nTarget:")
print("Pass")

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "logistic_regression",
        LogisticRegression(
            random_state=42
        )
    )
])


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# 10. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Student Pass/Fail - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Predicted Fail", "Predicted Pass"]
)

plt.yticks(
    [0, 1],
    ["Actual Fail", "Actual Pass"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "day15_confusion_matrix.png",
    dpi=300
)

plt.show()

print(
    "\nSaved: day15_confusion_matrix.png"
)


# ============================================================
# 11. CREATE PREDICTION RESULTS
# ============================================================

results = X_test.copy()

results["Actual_Pass"] = y_test.values

results["Predicted_Pass"] = y_pred

results["Pass_Probability"] = y_probability

results["Actual_Result"] = results[
    "Actual_Pass"
].map({
    0: "Fail",
    1: "Pass"
})

results["Predicted_Result"] = results[
    "Predicted_Pass"
].map({
    0: "Fail",
    1: "Pass"
})


# ============================================================
# 12. ANALYZE FALSE POSITIVES / NEGATIVES
# ============================================================

results["Prediction_Type"] = "Correct"

results.loc[
    (results["Actual_Pass"] == 0) &
    (results["Predicted_Pass"] == 1),
    "Prediction_Type"
] = "False Positive"

results.loc[
    (results["Actual_Pass"] == 1) &
    (results["Predicted_Pass"] == 0),
    "Prediction_Type"
] = "False Negative"


print("\nPrediction results:")
print(results)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

results.to_csv(
    "day15_predictions.csv",
    index=False
)

print(
    "\nSaved: day15_predictions.csv"
)


# ============================================================
# 14. BUSINESS / REAL-WORLD IMPLICATIONS
# ============================================================

print("\n" + "=" * 60)
print("REAL-WORLD ERROR ANALYSIS")
print("=" * 60)

print(
    "\nFalse Positive:"
)

print(
    "The model predicts that a student will pass, "
    "but the student actually fails."
)

print(
    "Impact: The teacher may fail to provide additional "
    "support to a student who needs it."
)

print(
    "\nFalse Negative:"
)

print(
    "The model predicts that a student will fail, "
    "but the student actually passes."
)

print(
    "Impact: The student may receive unnecessary "
    "intervention or additional academic support."
)


# ============================================================
# 15. SAVE RESULTS SUMMARY
# ============================================================

with open(
    "day15_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DAY 15 - LOGISTIC REGRESSION CLASSIFICATION\n"
    )

    file.write(
        "============================================\n\n"
    )

    file.write(
        "Feature: name_length\n"
    )

    file.write(
        "Target: Pass / Fail\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n\n"
    )

    file.write(
        f"True Negatives: {tn}\n"
    )

    file.write(
        f"False Positives: {fp}\n"
    )

    file.write(
        f"False Negatives: {fn}\n"
    )

    file.write(
        f"True Positives: {tp}\n\n"
    )

    file.write(
        "REAL-WORLD IMPLICATIONS\n"
    )

    file.write(
        "-----------------------\n"
    )

    file.write(
        "False positives can lead to unnecessary "
        "student interventions.\n"
    )

    file.write(
        "False negatives can cause students who need "
        "support to be overlooked.\n"
    )

print(
    "\nSaved: day15_results.txt"
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DAY 15 COMPLETED SUCCESSFULLY")
print("=" * 60)