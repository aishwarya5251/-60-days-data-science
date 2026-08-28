import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, f1_score
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


n_samples = 15000
fraud_rate = 0.034

rng = np.random.default_rng(RANDOM_STATE)

# --- Legitimate baseline transaction behavior ---
distance_from_home = rng.gamma(shape=2.0, scale=8.0, size=n_samples)
distance_from_last_transaction = rng.gamma(shape=1.2, scale=4.0, size=n_samples)
ratio_to_median_purchase_price = rng.gamma(shape=2.5, scale=0.6, size=n_samples)
transaction_amount = rng.lognormal(mean=3.6, sigma=1.0, size=n_samples)
transaction_hour = rng.integers(0, 24, size=n_samples)
repeat_retailer = rng.binomial(1, 0.85, size=n_samples)
used_chip = rng.binomial(1, 0.65, size=n_samples)
used_pin_number = rng.binomial(1, 0.45, size=n_samples)
online_order = rng.binomial(1, 0.35, size=n_samples)
num_transactions_last_24h = rng.poisson(lam=2.0, size=n_samples)
account_age_days = rng.integers(1, 3650, size=n_samples)

df = pd.DataFrame({
    "transaction_amount": transaction_amount,
    "distance_from_home": distance_from_home,
    "distance_from_last_transaction": distance_from_last_transaction,
    "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
    "transaction_hour": transaction_hour,
    "repeat_retailer": repeat_retailer,
    "used_chip": used_chip,
    "used_pin_number": used_pin_number,
    "online_order": online_order,
    "num_transactions_last_24h": num_transactions_last_24h,
    "account_age_days": account_age_days,
})

# --- Latent fraud-risk score drives the label (nonlinear + noisy) ---
risk_score = (
    0.9 * (df["ratio_to_median_purchase_price"] > 3.0).astype(int)
    + 0.8 * (df["distance_from_home"] > 25).astype(int)
    + 0.7 * (df["distance_from_last_transaction"] > 15).astype(int)
    + 0.6 * df["online_order"]
    - 0.9 * df["used_pin_number"]
    - 0.5 * df["used_chip"]
    - 0.4 * df["repeat_retailer"]
    + 0.5 * ((df["transaction_hour"] >= 0) & (df["transaction_hour"] <= 5)).astype(int)
    + 0.4 * (df["num_transactions_last_24h"] > 5).astype(int)
    + 0.3 * (df["account_age_days"] < 30).astype(int)
    + rng.normal(0, 0.9, size=n_samples)   # noise = real-world label ambiguity
)

threshold = np.quantile(risk_score, 1 - fraud_rate)
df["is_fraud"] = (risk_score >= threshold).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.3%}  ({df['is_fraud'].sum()} fraud / {len(df)} total)")
df.head()


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df["is_fraud"].value_counts().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"])
axes[0].set_title("Class Distribution")
axes[0].set_xticklabels(["Legitimate", "Fraud"], rotation=0)
axes[0].set_ylabel("Count")

sns.boxplot(data=df, x="is_fraud", y="ratio_to_median_purchase_price", ax=axes[1])
axes[1].set_title("Purchase Price Ratio by Class")
axes[1].set_xticklabels(["Legitimate", "Fraud"])

sns.boxplot(data=df, x="is_fraud", y="distance_from_home", ax=axes[2])
axes[2].set_title("Distance From Home by Class")
axes[2].set_xticklabels(["Legitimate", "Fraud"])

plt.tight_layout()
plt.savefig("figures/eda_overview.png", bbox_inches="tight")
plt.show()


plt.figure(figsize=(9, 7))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", bbox_inches="tight")
plt.show()


X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape}, fraud rate = {y_train.mean():.3%}")
print(f"Test:  {X_test.shape}, fraud rate = {y_test.mean():.3%}")


dt = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_proba = dt.predict_proba(X_test)[:, 1]

print("Decision Tree — Test Set Performance")
print(classification_report(y_test, dt_pred, target_names=["Legitimate", "Fraud"], digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, dt_proba):.4f}")
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, dt_proba):.4f}")


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("Random Forest — Test Set Performance")
print(classification_report(y_test, rf_pred, target_names=["Legitimate", "Fraud"], digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, rf_proba):.4f}")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, pred) in zip(axes, [("Decision Tree", dt_pred), ("Random Forest", rf_pred)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curves
for name, proba in [("Decision Tree", dt_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Precision-Recall curves
for name, proba in [("Decision Tree", dt_proba), ("Random Forest", rf_proba)]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(rec, prec, label=f"{name} (AP = {ap:.3f})")
axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.4, label="Baseline (fraud rate)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", bbox_inches="tight")
plt.show()


summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (fraud)", "Recall (fraud)", "F1 (fraud)", "ROC-AUC", "PR-AUC"],
    "Decision Tree": [
        (dt_pred == y_test).mean(),
        classification_report(y_test, dt_pred, output_dict=True)["1"]["precision"],
        classification_report(y_test, dt_pred, output_dict=True)["1"]["recall"],
        classification_report(y_test, dt_pred, output_dict=True)["1"]["f1-score"],
        roc_auc_score(y_test, dt_proba),
        average_precision_score(y_test, dt_proba),
    ],
    "Random Forest": [
        (rf_pred == y_test).mean(),
        classification_report(y_test, rf_pred, output_dict=True)["1"]["precision"],
        classification_report(y_test, rf_pred, output_dict=True)["1"]["recall"],
        classification_report(y_test, rf_pred, output_dict=True)["1"]["f1-score"],
        roc_auc_score(y_test, rf_proba),
        average_precision_score(y_test, rf_proba),
    ],
})
summary["Improvement"] = summary["Random Forest"] - summary["Decision Tree"]
summary.round(4)


dt_importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

dt_importance.plot(kind="barh", ax=axes[0], color="#C44E52")
axes[0].invert_yaxis()
axes[0].set_title("Decision Tree — Feature Importance")
axes[0].set_xlabel("Importance (MDI)")

rf_importance.plot(kind="barh", ax=axes[1], color="#4C72B0")
axes[1].invert_yaxis()
axes[1].set_title("Random Forest — Feature Importance")
axes[1].set_xlabel("Importance (MDI)")

plt.tight_layout()
plt.savefig("figures/feature_importance.png", bbox_inches="tight")
plt.show()

importance_table = pd.DataFrame({
    "Decision Tree": dt_importance,
    "Random Forest": rf_importance,
}).sort_values("Random Forest", ascending=False)
importance_table.round(4)


# Permutation importance is a more reliable alternative to MDI: it measures the
# actual drop in performance when a feature's values are shuffled, rather than
# relying on impurity reduction (which is biased toward high-cardinality features).
from sklearn.inspection import permutation_importance

perm = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=RANDOM_STATE, scoring="f1")
perm_importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
perm_importance.plot(kind="barh", color="#55A868")
plt.gca().invert_yaxis()
plt.title("Random Forest — Permutation Importance (F1 drop)")
plt.xlabel("Mean F1 decrease when feature is shuffled")
plt.tight_layout()
plt.savefig("figures/permutation_importance.png", bbox_inches="tight")
plt.show()

perm_importance.round(4)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = ["f1", "roc_auc", "average_precision"]

dt_cv = cross_validate(dt, X, y, cv=cv, scoring=scoring)
rf_cv = cross_validate(rf, X, y, cv=cv, scoring=scoring)

cv_summary = pd.DataFrame({
    "Decision Tree (mean)": [dt_cv[f"test_{s}"].mean() for s in scoring],
    "Decision Tree (std)": [dt_cv[f"test_{s}"].std() for s in scoring],
    "Random Forest (mean)": [rf_cv[f"test_{s}"].mean() for s in scoring],
    "Random Forest (std)": [rf_cv[f"test_{s}"].std() for s in scoring],
}, index=["F1", "ROC-AUC", "PR-AUC"])

cv_summary.round(4)


# Lower std across folds = more robust/stable model.
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(scoring))
width = 0.35

ax.bar(x - width/2, [dt_cv[f"test_{s}"].mean() for s in scoring],
       width, yerr=[dt_cv[f"test_{s}"].std() for s in scoring],
       label="Decision Tree", color="#C44E52", capsize=5)
ax.bar(x + width/2, [rf_cv[f"test_{s}"].mean() for s in scoring],
       width, yerr=[rf_cv[f"test_{s}"].std() for s in scoring],
       label="Random Forest", color="#4C72B0", capsize=5)

ax.set_xticks(x)
ax.set_xticklabels(["F1", "ROC-AUC", "PR-AUC"])
ax.set_ylabel("Score")
ax.set_title("5-Fold Cross-Validation: Mean ± Std")
ax.legend()
plt.tight_layout()
plt.savefig("figures/cv_robustness.png", bbox_inches="tight")
plt.show()


# Sensitivity to forest size
tree_counts = [5, 10, 25, 50, 100, 200, 300, 500]
f1_scores = []

for n in tree_counts:
    model = RandomForestClassifier(
        n_estimators=n, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    f1_scores.append(f1_score(y_test, model.predict(X_test)))

plt.figure(figsize=(7, 4.5))
plt.plot(tree_counts, f1_scores, marker="o", color="#4C72B0")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("F1 Score (fraud class)")
plt.title("Random Forest: Performance vs. Forest Size")
plt.tight_layout()
plt.savefig("figures/forest_size_sensitivity.png", bbox_inches="tight")
plt.show()

print("F1 by forest size:", dict(zip(tree_counts, np.round(f1_scores, 4))))


# Robustness to feature noise: add Gaussian noise to continuous features at
# increasing intensity and see how much each model's F1 degrades.
noisy_cols = ["transaction_amount", "distance_from_home",
              "distance_from_last_transaction", "ratio_to_median_purchase_price"]

noise_levels = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
dt_noise_f1, rf_noise_f1 = [], []

for level in noise_levels:
    X_noisy = X_test.copy()
    for col in noisy_cols:
        noise = rng.normal(0, level * X_test[col].std(), size=len(X_test))
        X_noisy[col] = (X_test[col] + noise).clip(lower=0)

    dt_noise_f1.append(f1_score(y_test, dt.predict(X_noisy)))
    rf_noise_f1.append(f1_score(y_test, rf.predict(X_noisy)))

plt.figure(figsize=(7, 4.5))
plt.plot(noise_levels, dt_noise_f1, marker="o", label="Decision Tree", color="#C44E52")
plt.plot(noise_levels, rf_noise_f1, marker="o", label="Random Forest", color="#4C72B0")
plt.xlabel("Injected Noise Level (fraction of feature std-dev)")
plt.ylabel("F1 Score (fraud class)")
plt.title("Robustness to Feature Noise")
plt.legend()
plt.tight_layout()
plt.savefig("figures/noise_robustness.png", bbox_inches="tight")
plt.show()

print("F1 under noise (Decision Tree):", dict(zip(noise_levels, np.round(dt_noise_f1, 4))))
print("F1 under noise (Random Forest):", dict(zip(noise_levels, np.round(rf_noise_f1, 4))))


# Save trained artifacts + processed dataset for reuse / repo submission
import joblib

df.to_csv("data/synthetic_fraud_transactions.csv", index=False)
joblib.dump(dt, "reports/decision_tree_model.pkl")
joblib.dump(rf, "reports/random_forest_model.pkl")
summary.round(4).to_csv("reports/performance_comparison.csv", index=False)
importance_table.round(4).to_csv("reports/feature_importance.csv")

print("Saved dataset, models, and reports.")