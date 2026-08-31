# ============================================
# DAY 20 - MODEL EVALUATION
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# 1. Load Dataset
df = pd.read_csv("students.csv")

print("Dataset:")
print(df)

print("\nDataset Shape:", df.shape)
# 2. Create Classification Target
# G3 >= 10 = Pass
# G3 < 10 = Fail

df["target"] = (df["G3"] >= 10).astype(int)

print("\nTarget:")
print(df["target"].value_counts())
# 3. Convert Grade into Numbers
encoder = LabelEncoder()

df["Grade"] = encoder.fit_transform(df["Grade"])

print("\nEncoded Dataset:")
print(df.head())
# 4. Select Features and Target
# Name is not useful for prediction
X = df[["Grade", "G3"]]

y = df["target"]
# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# 6. Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 7. Create Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}
# 8. Train and Evaluate Models
results = {}

for name, model in models.items():

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
    # Print results
    print("\n================================")
    print(name)
    print("================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))
# 9. Metrics Comparison Table
results_df = pd.DataFrame(
    results,
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

print("\n\nMODEL COMPARISON")
print(results_df.round(4))
# 10. Confusion Matrices
for name, model in models.items():

    if name == "Logistic Regression":
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"]
    )

    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
# 11. Best Model
best_model = results_df.loc["F1 Score"].idxmax()

print("\n================================")
print("BEST MODEL")
print("================================")

print("Best Model:", best_model)
print(
    "F1 Score:",
    round(results_df.loc["F1 Score", best_model], 4)
)