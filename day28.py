import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("students.csv")

df["target"] = (df["G3"] >= 10).astype(int)

X = df[["Grade"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("grade", OneHotEncoder(handle_unknown="ignore"), ["Grade"])
    ]
)

base_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

base_pipeline.fit(X_train, y_train)

base_pred = base_pipeline.predict(X_test)

base_accuracy = accuracy_score(y_test, base_pred)
base_precision = precision_score(y_test, base_pred, zero_division=0)
base_recall = recall_score(y_test, base_pred, zero_division=0)
base_f1 = f1_score(y_test, base_pred, zero_division=0)

print("BASE PIPELINE")
print("Accuracy:", round(base_accuracy, 4))
print("Precision:", round(base_precision, 4))
print("Recall:", round(base_recall, 4))
print("F1 Score:", round(base_f1, 4))

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [2, 3, 5, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_

final_pred = best_pipeline.predict(X_test)

final_accuracy = accuracy_score(y_test, final_pred)
final_precision = precision_score(y_test, final_pred, zero_division=0)
final_recall = recall_score(y_test, final_pred, zero_division=0)
final_f1 = f1_score(y_test, final_pred, zero_division=0)

cv_scores = cross_val_score(
    best_pipeline,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION F1:",
      round(grid_search.best_score_, 4))

print("\nFINAL PIPELINE")
print("Accuracy:", round(final_accuracy, 4))
print("Precision:", round(final_precision, 4))
print("Recall:", round(final_recall, 4))
print("F1 Score:", round(final_f1, 4))

print("\nFINAL CROSS-VALIDATION")
print("Mean F1:", round(cv_scores.mean(), 4))
print("Standard Deviation:", round(cv_scores.std(), 4))

comparison = pd.DataFrame({
    "Model": [
        "Base Pipeline",
        "Optimized Pipeline"
    ],
    "Accuracy": [
        base_accuracy,
        final_accuracy
    ],
    "Precision": [
        base_precision,
        final_precision
    ],
    "Recall": [
        base_recall,
        final_recall
    ],
    "F1 Score": [
        base_f1,
        final_f1
    ]
})

print("\nPERFORMANCE COMPARISON")
print(comparison.round(4))

comparison.to_csv(
    "day28_final_performance.csv",
    index=False
)

parameters = pd.DataFrame([
    grid_search.best_params_
])

parameters.to_csv(
    "day28_best_parameters.csv",
    index=False
)

plt.figure(figsize=(8, 5))

comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar")

plt.title("Base vs Optimized Pipeline")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nDay 28 Optimized ML Pipeline completed successfully!")