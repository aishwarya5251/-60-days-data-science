import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

df["target"] = (df["G3"] >= 10).astype(int)

encoder = LabelEncoder()
df["Grade"] = encoder.fit_transform(df["Grade"])

X = df[["Grade"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

pred_before = model.predict(X_test)

accuracy_before = accuracy_score(
    y_test,
    pred_before
)

print("Accuracy Before Tuning:",
      round(accuracy_before, 4))

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

pred_after = best_model.predict(X_test)

accuracy_after = accuracy_score(
    y_test,
    pred_after
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:",
      round(grid_search.best_score_, 4))

print("\nAccuracy After Tuning:",
      round(accuracy_after, 4))

comparison = pd.DataFrame({
    "Model": [
        "Random Forest Before Tuning",
        "Random Forest After Tuning"
    ],
    "Accuracy": [
        accuracy_before,
        accuracy_after
    ]
})

print("\nPERFORMANCE COMPARISON")
print(comparison.round(4))

comparison.to_csv(
    "day26_tuning_comparison.csv",
    index=False
)

best_parameters = pd.DataFrame([
    grid_search.best_params_
])

best_parameters.to_csv(
    "day26_best_parameters.csv",
    index=False
)

print("\nDay 26 Hyperparameter Tuning completed successfully!")
