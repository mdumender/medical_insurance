import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

# ===============================
# VERİ
# ===============================
df = pd.read_csv(r"C:\Users\MERT\Downloads\medical_insurance.csv")

df.columns = df.columns.str.lower().str.strip()
df.rename(columns={'gender': 'sex', 'expenses': 'charges'}, inplace=True)

X = df[['age', 'bmi', 'children', 'sex', 'region', 'discount_eligibility']]
y = df['charges']

num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'region', 'discount_eligibility']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first'), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODELLER
# ===============================
models = {
    "RandomForest": (
        Pipeline([('prep', preprocess), ('rf', RandomForestRegressor(random_state=42))]),
        {'rf__n_estimators': [300, 500], 'rf__max_depth': [5, 10, None]}
    ),
    "XGBoost": (
        Pipeline([('prep', preprocess), ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))]),
        {'xgb__n_estimators': [300, 500], 'xgb__max_depth': [3, 5], 'xgb__learning_rate': [0.05, 0.1]}
    ),
    "KNN": (
        Pipeline([('prep', preprocess), ('knn', KNeighborsRegressor())]),
        {'knn__n_neighbors': [5, 10, 20], 'knn__weights': ['distance']}
    )
}

best_model = None
best_mae = np.inf
best_name = None

for name, (pipe, param_grid) in models.items():
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid.fit(X_train, y_train)

    preds = grid.best_estimator_.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"{name} MAE: {mae:.2f} | Best Params: {grid.best_params_}")

    if mae < best_mae:
        best_mae = mae
        best_model = grid.best_estimator_
        best_name = name

# ===============================
# EN İYİ MODELİ KAYDET
# ===============================
joblib.dump(best_model, "best_pricing_model.pkl")

print(f"\nEn iyi model: {best_name}")
print(f"MAE: {best_mae:.2f}")
print("Model kaydedildi: best_pricing_model.pkl")
