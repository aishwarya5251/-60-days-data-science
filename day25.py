import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

split_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    split_results.append({
        "Model": name,
        "Train-Test Accuracy": accuracy
    })

split_df = pd.DataFrame(split_results)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    cv_results.append({
        "Model": name,
        "CV Mean Accuracy": scores.mean(),
        "CV Std": scores.std(),
        "Minimum CV Score": scores.min(),
        "Maximum CV Score": scores.max()
    })

cv_df = pd.DataFrame(cv_results)

comparison = split_df.merge(
    cv_df,
    on="Model"
)

print("\nTRAIN-TEST SPLIT RESULTS")
print(split_df.round(4))

print("\nCROSS-VALIDATION RESULTS")
print(cv_df.round(4))

print("\nFINAL COMPARISON")
print(comparison.round(4))

best_model = cv_df.loc[
    cv_df["CV Mean Accuracy"].idxmax()
]

print("\nBEST MODEL:")
print(best_model["Model"])

print(
    "Mean CV Accuracy:",
    round(best_model["CV Mean Accuracy"], 4)
)

print(
    "CV Standard Deviation:",
    round(best_model["CV Std"], 4)
)

comparison.to_csv(
    "day25_cross_validation_comparison.csv",
    index=False
)

print("\nDay 25 Cross-Validation completed successfully!")