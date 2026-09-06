import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

print("Original Dataset")
print(df.head())

df["target"] = (df["G3"] >= 10).astype(int)

encoder = LabelEncoder()
df["Grade"] = encoder.fit_transform(df["Grade"])

X = df[["Grade", "G3"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_before = LogisticRegression()

model_before.fit(X_train_scaled, y_train)

pred_before = model_before.predict(X_test_scaled)

accuracy_before = accuracy_score(
    y_test,
    pred_before
)

print("\nAccuracy Before PCA:",
      round(accuracy_before, 4))

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:",
      round(pca.explained_variance_ratio_.sum(), 4))

model_after = LogisticRegression()

model_after.fit(X_train_pca, y_train)

pred_after = model_after.predict(X_test_pca)

accuracy_after = accuracy_score(
    y_test,
    pred_after
)

print("\nAccuracy After PCA:",
      round(accuracy_after, 4))

pca_df = pd.DataFrame(
    X_train_pca,
    columns=["PC1", "PC2"]
)

pca_df["Target"] = y_train.values

plt.figure(figsize=(8, 5))

for label in pca_df["Target"].unique():
    data = pca_df[pca_df["Target"] == label]

    plt.scatter(
        data["PC1"],
        data["PC2"],
        label="Pass" if label == 1 else "Fail"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization")
plt.legend()
plt.show()

variance = pd.DataFrame({
    "Component": ["PC1", "PC2"],
    "Explained Variance": pca.explained_variance_ratio_
})

print("\nVariance Analysis")
print(variance)

comparison = pd.DataFrame({
    "Method": [
        "Without PCA",
        "With PCA"
    ],
    "Accuracy": [
        accuracy_before,
        accuracy_after
    ]
})

print("\nPerformance Comparison")
print(comparison.round(4))

comparison.to_csv(
    "day24_pca_comparison.csv",
    index=False
)

variance.to_csv(
    "day24_explained_variance.csv",
    index=False
)

print("\nDay 24 PCA analysis completed successfully!")