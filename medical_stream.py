import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Health Insurance Pricing Demo", layout="centered")


# =========================================================
# MODEL LOAD / TRAIN (Cloud-safe)
# =========================================================
@st.cache_resource
def load_or_train_model():

    model_path = "final_rf_model.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)

    # Model yoksa eğit
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

    joblib.dump(model, model_path)

    return model


model = load_or_train_model()


# =========================================================
# UI
# =========================================================
st.title("Health Insurance – Expected Cost Estimator")
st.markdown("Bu uygulama bireysel poliçeler için **beklenen sağlık maliyetini (pure premium)** tahmin eder.")

st.sidebar.header("Poliçe Bilgileri")

age = st.sidebar.number_input("Age", 18, 100, 40)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 27.5)
children = st.sidebar.number_input("Children", 0, 5, 1)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
discount_eligibility = st.sidebar.selectbox("Discount Eligibility", ["yes", "no"])

expense_loading = st.sidebar.slider("Expense Loading (%)", 0, 40, 15) / 100
profit_margin = st.sidebar.slider("Profit Margin (%)", 0, 40, 10) / 100


input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex": sex,
    "region": region,
    "discount_eligibility": discount_eligibility
}])


if st.button("Beklenen Maliyeti Hesapla"):

    expected_cost = model.predict(input_df)[0]
    gross_premium = expected_cost * (1 + expense_loading + profit_margin)

    st.subheader("Sonuçlar")
    st.metric("Beklenen Yıllık Hasar (Pure Premium)", f"{expected_cost:,.2f} ₺")
    st.metric("Önerilen Brüt Prim (Simülasyon)", f"{gross_premium:,.2f} ₺")

    st.info("""
    Bu çıktı teknik primdir (expected loss).
    Brüt prim; şirketin masraf yapısı, sermaye maliyeti ve hedef kârlılığına göre değişir.
    """)


st.markdown("---")


# =========================================================
# ANALİTİK PANEL
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("medical_insurance.csv")
    df.columns = df.columns.str.lower().str.strip()
    return df


df = load_data()

st.markdown("## Portföy Analizi (Gerçekleşen Hasarlar)")


# --- Cinsiyet ---
sex_summary = df.groupby("sex")["charges"].mean().reset_index()

st.subheader("Cinsiyet Bazında Ortalama Hasar")
st.dataframe(sex_summary)
st.bar_chart(sex_summary.set_index("sex"))


# --- Cohort ---
bins = [18, 30, 40, 50, 60, 100]
labels = ["18-30", "31-40", "41-50", "51-60", "60+"]

df["cohort"] = pd.cut(df["age"], bins=bins, labels=labels)

cohort_summary = df.groupby("cohort")["charges"].mean().reset_index()

st.subheader("Cohort (Yaş Grubu) Bazında Ortalama Hasar")
st.dataframe(cohort_summary)
st.bar_chart(cohort_summary.set_index("cohort"))


# =========================================================
# ACTUAL VS EXPECTED
# =========================================================
st.markdown("## Actual vs Expected (Random Forest – Test Set)")

@st.cache_data
def load_actual_expected():

    if os.path.exists("actual_vs_predicted_test.xlsx"):
        return pd.read_excel("actual_vs_predicted_test.xlsx")
    else:
        return None


ae_df = load_actual_expected()

if ae_df is not None:

    ae_df.columns = ae_df.columns.str.lower().str.strip()

    st.dataframe(ae_df.head(20))

    fig = px.scatter(
        ae_df,
        x="actual",
        y="expected",
        title="Actual vs Expected",
        opacity=0.6
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("actual_vs_predicted_test.xlsx dosyası repo içinde bulunamadı.")


st.markdown("---")
st.caption("Mert DÜMENDER Actuarial Pricing Demo – MAD@2026")
