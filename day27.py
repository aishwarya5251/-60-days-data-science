import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

df["target"] = (df["G3"] >= 10).astype(int)

encoder = LabelEncoder()
df["Grade"] = encoder.fit_transform(df["Grade"])

X = df[["Grade"]]
y = df["target"]

models = {
    "Low Complexity": RandomForestClassifier(
        n_estimators=50,
        max_depth=2,
        random_state=42
    ),
    "Medium Complexity": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    "High Complexity": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, model in models.items():
    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5)
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    results.append({
        "Model": name,
        "Training Accuracy": train_mean[-1],
        "Validation Accuracy": validation_mean[-1],
        "Gap": train_mean[-1] - validation_mean[-1]
    })

    plt.figure(figsize=(8, 5))
    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training Accuracy"
    )
    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()

comparison = pd.DataFrame(results)

print("\nBIAS-VARIANCE ANALYSIS")
print(comparison.round(4))

comparison.to_csv(
    "day27_bias_variance_comparison.csv",
    index=False
)

print("\nDay 27 Bias-Variance Analysis completed successfully!")