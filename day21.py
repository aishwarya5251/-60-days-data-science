# ============================================
# DAY 21 - SPRINT REVIEW & MODEL SELECTION
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
    roc_auc_score
)
# 1. Load Dataset===
df = pd.read_csv("students.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
# 2. Create Target Variable
# G3 >= 10 = Pass
# G3 < 10 = Fail

df["target"] = (df["G3"] >= 10).astype(int)
# 3. Encode Grade
encoder = LabelEncoder()

df["Grade"] = encoder.fit_transform(df["Grade"])
# 4. Select Features
# Do NOT use G3 because it was used to create target.
# This prevents data leakage.

X = df[["Grade"]]
y = df["target"]
# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# 6. Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 7. Classification Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}
# 8. Evaluate Models
results = []

for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })
# 9. Comparison Table
results_df = pd.DataFrame(results)

print("\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(results_df.round(4))
# 10. Find Best Model

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))
print("ROC-AUC:", round(best_model["ROC-AUC"], 4))
# 11. Visual Comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Classification Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()
# 12. Final Conclusion

print("\n========================================")
print("FINAL CONCLUSION")
print("========================================")

print(
    f"{best_model['Model']} was selected as the best model "
    f"based on the highest F1 Score."
)

print(
    "The model provides a balance between precision and recall "
    "and is therefore suitable for the student pass/fail "
    "classification problem."
)