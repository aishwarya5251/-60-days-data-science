import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("students.csv")

print("ORIGINAL DATASET")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

df["target"] = (df["G3"] >= 10).astype(int)

print("\nTarget Distribution:")
print(df["target"].value_counts())

label_df = df.copy()

label_encoder = LabelEncoder()

label_df["Grade"] = label_encoder.fit_transform(
    label_df["Grade"]
)

print("\nLABEL ENCODED DATA")
print(label_df.head())

print("\nGrade Mapping:")

for value, encoded in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(value, "=", encoded)

onehot_df = df.copy()

onehot_df = pd.get_dummies(
    onehot_df,
    columns=["Grade"],
    dtype=int
)

print("\nONE-HOT ENCODED DATA")
print(onehot_df.head())

print("\nOne-Hot Columns:")
print(onehot_df.columns.tolist())

print("\nSTRUCTURE COMPARISON")
print("Original Shape:", df.shape)
print("Label Encoding Shape:", label_df.shape)
print("One-Hot Encoding Shape:", onehot_df.shape)

X_label = label_df[["Grade"]]
y = label_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_label,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_label = LogisticRegression()

model_label.fit(X_train, y_train)

pred_label = model_label.predict(X_test)

accuracy_label = accuracy_score(
    y_test,
    pred_label
)

print("\nLabel Encoding Accuracy:",
      round(accuracy_label, 4))

X_onehot = onehot_df[
    ["Grade_A", "Grade_B", "Grade_C", "Grade_D"]
]

X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(
    X_onehot,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model_onehot = LogisticRegression()

model_onehot.fit(X_train_oh, y_train_oh)

pred_onehot = model_onehot.predict(X_test_oh)

accuracy_onehot = accuracy_score(
    y_test_oh,
    pred_onehot
)

print("One-Hot Encoding Accuracy:",
      round(accuracy_onehot, 4))

comparison = pd.DataFrame({
    "Encoding": [
        "Label Encoding",
        "One-Hot Encoding"
    ],
    "Accuracy": [
        accuracy_label,
        accuracy_onehot
    ]
})

print("\nPERFORMANCE COMPARISON")
print(comparison.round(4))

label_df.to_csv(
    "students_label_encoded.csv",
    index=False
)

onehot_df.to_csv(
    "students_onehot_encoded.csv",
    index=False
)

print("\nEncoded datasets saved successfully!")