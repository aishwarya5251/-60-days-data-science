import xgboost
print(xgboost.__version__)
# Day 19 - Boosting Model Performance with XGBoost

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# -----------------------------
# 1. Load Datas
df = pd.read_csv("students.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# -----------------------------
# 2. Prepare Data
# -----------------------------

# Example: target variable is G3
X = df.drop("G3", axis=1)
y = df["G3"]

# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# 3. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# -----------------------------
# 4. Random Forest Model
# -----------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

# -----------------------------
# 5. XGBoost Model
# -----------------------------

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)

# -----------------------------
# 6. Performance Comparison
# -----------------------------

comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "MAE": [rf_mae, xgb_mae],
    "RMSE": [rf_rmse, xgb_rmse],
    "R2 Score": [rf_r2, xgb_r2]
})

print("\nModel Performance Comparison:")
print(comparison)
df = pd.read_csv("students.csv")
X = df.drop("G3", axis=1)
y = df["G3"]