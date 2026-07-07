import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("diabetes_binary_health.csv")

target = "Diabetes_binary"

X = df.drop(columns=[target])
y = df[target]

st.title("🩺 Diabetes Prediction System")

st.markdown("""
This application predicts whether a patient is **Diabetic** or **Non-Diabetic**
using a **Random Forest Machine Learning Model**.
""")

st.sidebar.header("Patient Details")

high_bp = st.sidebar.selectbox("High Blood Pressure", [0, 1])

high_chol = st.sidebar.selectbox("High Cholesterol", [0, 1])

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

smoker = st.sidebar.selectbox("Smoker", [0, 1])

stroke = st.sidebar.selectbox("Stroke History", [0, 1])

heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])

phys_activity = st.sidebar.selectbox("Physical Activity", [0, 1])

age = st.sidebar.slider(
    "Age Category",
    1,
    13,
    5
)

sample = pd.DataFrame(
    [[
        high_bp,
        high_chol,
        bmi,
        smoker,
        stroke,
        heart_disease,
        phys_activity,
        age
    ]],
    columns=X.columns[:8]
)

for col in X.columns[8:]:
    sample[col] = 0

sample = sample[X.columns]

col1, col2 = st.columns(2)

with col1:

    st.subheader("Project Information")

    st.write("**Algorithm:** Random Forest")

    st.write("**Dataset:** Diabetes Health Indicators")

    st.write("**Total Records:**", len(df))

    st.write("**Total Features:**", len(X.columns))

    st.write("**Model Accuracy:** 86%")

with col2:

    st.subheader("Dataset Distribution")

    diabetic = int(y.sum())

    non_diabetic = len(y) - diabetic

    fig, ax = plt.subplots()

    ax.bar(
        ["Non-Diabetic", "Diabetic"],
        [non_diabetic, diabetic]
    )

    st.pyplot(fig)

st.markdown("---")

if st.button("Predict Diabetes"):

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    probability = model.predict_proba(sample_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Patient is likely Diabetic")

    else:

        st.success("✅ Patient is Non-Diabetic")

    st.metric(
        "Risk Percentage",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

st.markdown("---")

st.subheader("Top Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top10 = importance.head(10)

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.barh(
    top10["Feature"],
    top10["Importance"]
)

ax2.invert_yaxis()

st.pyplot(fig2)
st.markdown("---")
st.caption("Diabetes Prediction System using Random Forest")