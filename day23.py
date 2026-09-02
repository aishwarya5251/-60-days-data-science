import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

print("DATASET")
print(df.head())

print("\nDataset Shape:", df.shape)

df["target"] = (df["G3"] >= 10).astype(int)

encoder = LabelEncoder()
df["Grade"] = encoder.fit_transform(df["Grade"])

print("\nCorrelation Matrix:")
print(df[["Grade", "G3", "target"]].corr())

plt.figure(figsize=(6, 4))
sns.heatmap(
    df[["Grade", "G3", "target"]].corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Feature Correlation")
plt.show()

X = df[["Grade", "G3"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_before = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_before.fit(X_train, y_train)

pred_before = model_before.predict(X_test)

accuracy_before = accuracy_score(
    y_test,
    pred_before
)

print("\nAccuracy Before Feature Selection:",
      round(accuracy_before, 4))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model_before.feature_importances_
})

print("\nFeature Importance:")
print(importance.sort_values(
    by="Importance",
    ascending=False
))

plt.figure(figsize=(7, 4))

sns.barplot(
    data=importance.sort_values(
        by="Importance",
        ascending=False
    ),
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.show()

selected_features = ["Grade"]

X_selected = df[selected_features]

X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_after = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_after.fit(
    X_train_selected,
    y_train_selected
)

pred_after = model_after.predict(X_test_selected)

accuracy_after = accuracy_score(
    y_test_selected,
    pred_after
)

print("\nAccuracy After Feature Selection:",
      round(accuracy_after, 4))

comparison = pd.DataFrame({
    "Stage": [
        "Before Feature Selection",
        "After Feature Selection"
    ],
    "Accuracy": [
        accuracy_before,
        accuracy_after
    ]
})

print("\nPERFORMANCE COMPARISON")
print(comparison.round(4))

comparison.to_csv(
    "day23_feature_selection_comparison.csv",
    index=False
)

print("\nFeature selection analysis completed!")