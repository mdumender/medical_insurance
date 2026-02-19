import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Health Insurance Pricing Demo", layout="centered")


@st.cache_resource
def load_or_train_model():

    model_path = "final_rf_model.pkl"

    # Eğer model dosyası varsa yükle
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model

    # Yoksa modeli eğit
    df = pd.read_csv("medical_insurance.csv")

    X = df.drop("charges", axis=1)
    y = df["charges"]

    categorical_cols = ["sex", "region", "discount_eligibility"]
    numeric_cols = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    # Modeli kaydet
    joblib.dump(model, model_path)

    return model


model = load_or_train_model()
