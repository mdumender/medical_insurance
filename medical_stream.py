import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# Modeli yükle (GridSearch sonrası kaydettiğini varsayıyorum)
# ----------------------------------
model = joblib.load("final_rf_model.pkl")

st.set_page_config(page_title="Health Insurance Pricing Demo", layout="centered")

st.title("Health Insurance – Expected Cost Estimator")
st.markdown("Bu uygulama bireysel poliçeler için **beklenen sağlık maliyetini (pure premium)** tahmin eder.")
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Mert DÜMENDER Actuarial Pricing Demo – MAD@2026")
# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.header("Poliçe Bilgileri")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=40)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5)
children = st.sidebar.number_input("Children", min_value=0, max_value=5, value=1)
sex = st.sidebar.selectbox("Gender", ["male", "female"])
region = st.sidebar.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
discount_eligibility = st.sidebar.selectbox("Discount Eligibility", ["yes", "no"])

expense_loading = st.sidebar.slider("Expense Loading (%)", 0, 40, 15) / 100
profit_margin = st.sidebar.slider("Profit Margin (%)", 0, 40, 10) / 100

# ----------------------------------
# Prediction
# ----------------------------------
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

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Mert DÜMENDER Actuarial Pricing Demo – MAD@2026")

# ----------------------------------
# ANALİTİK PANELLER (COHORT & CİNSİYET)
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\MERT\Downloads\medical_insurance.csv")
    df.columns = df.columns.str.lower().str.strip()
    df.rename(columns={'gender': 'sex', 'expenses': 'charges'}, inplace=True)
    return df

df = load_data()

st.markdown("## Portföy Analizi (Gerçekleşen Hasarlar)")

# --- Cinsiyet Bazlı ---
sex_summary = df.groupby("sex")["charges"].mean().reset_index()
st.subheader("Cinsiyet Bazında Ortalama Hasar")
st.dataframe(sex_summary)

st.bar_chart(sex_summary.set_index("sex"))

# --- Cohort (Yaş Grubu) ---
bins = [18, 30, 40, 50, 60, 100]
labels = ["18-30", "31-40", "41-50", "51-60", "60+"]

df["cohort"] = pd.cut(df["age"], bins=bins, labels=labels)

cohort_summary = df.groupby("cohort")["charges"].mean().reset_index()
st.subheader("Cohort (Yaş Grubu) Bazında Ortalama Hasar")
st.dataframe(cohort_summary)

st.bar_chart(cohort_summary.set_index("cohort"))




import plotly.express as px

st.markdown("## 📈 Actual vs Expected (Random Forest – Test Set)")

@st.cache_data
def load_actual_expected():
    return pd.read_excel(r"C:\Users\MERT\Desktop\actual_vs_predicted_test.xlsx")

ae_df = load_actual_expected()

# Kolonları temizle
ae_df.columns = ae_df.columns.str.lower().str.strip()

# Gerekirse rename et (Excel'deki gerçek kolon isimlerine göre ayarla)
ae_df = ae_df.rename(columns={
    "actual": "actual",
    "expected": "expected"
})

st.write("Kolonlar:", ae_df.columns.tolist())
st.dataframe(ae_df.head(20))



