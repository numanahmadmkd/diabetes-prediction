# ─────────────────────────────────────
# Page configuration — MUST BE FIRST
# ─────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title = "Diabetes Prediction System",
    page_icon  = "🏥",
    layout     = "centered")

# ─────────────────────────────────────
# Import remaining libraries
# ─────────────────────────────────────
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# Load saved model and scaler
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

model  = load_model()
scaler = load_scaler()

# ─────────────────────────────────────
# Title and description
# ─────────────────────────────────────
st.title("🏥 Diabetes Prediction System")
st.markdown("#### Using XGBoost + SHAP Explainable AI")
st.markdown("---")
st.markdown("""
This system predicts whether a patient is at risk of diabetes
based on medical measurements. It also explains **why** the 
prediction was made using SHAP values.
""")

# ─────────────────────────────────────
# Input form
# ─────────────────────────────────────
st.markdown("## 📋 Enter Patient Information")
st.markdown("Please fill in all the medical values below:")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value = 0,
        max_value = 17,
        value     = 3,
        step      = 1,
        help      = "Number of times pregnant")

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value = 44,
        max_value = 200,
        value     = 120,
        step      = 1,
        help      = "Plasma glucose concentration")

    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value = 24,
        max_value = 122,
        value     = 70,
        step      = 1,
        help      = "Diastolic blood pressure")

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value = 7,
        max_value = 99,
        value     = 20,
        step      = 1,
        help      = "Triceps skin fold thickness")

with col2:
    insulin = st.number_input(
        "Insulin (μU/ml)",
        min_value = 14,
        max_value = 846,
        value     = 80,
        step      = 1,
        help      = "2-Hour serum insulin level")

    bmi = st.number_input(
        "BMI",
        min_value = 18.0,
        max_value = 67.0,
        value     = 32.0,
        step      = 0.1,
        help      = "Body Mass Index (weight/height²)")

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value = 0.07,
        max_value = 2.42,
        value     = 0.47,
        step      = 0.01,
        help      = "Diabetes family history score")

    age = st.number_input(
        "Age (years)",
        min_value = 21,
        max_value = 81,
        value     = 33,
        step      = 1,
        help      = "Patient age in years")

# ─────────────────────────────────────
# Predict button
# ─────────────────────────────────────
st.markdown("---")
predict_button = st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width = True)

# ─────────────────────────────────────
# Prediction and Results
# ─────────────────────────────────────
if predict_button:

    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi,
                            diabetes_pedigree, age]])

    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure',
                     'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']

    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]
    risk_percent = probability[1] * 100

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    # Show result
    if prediction == 1:
        st.error(f"""
        ### ⚠️ HIGH RISK — Diabetes Likely
        **Confidence: {risk_percent:.1f}%**
        
        This patient shows high risk of diabetes.
        Please consult a doctor immediately.
        """)
    else:
        st.success(f"""
        ### ✅ LOW RISK — Diabetes Unlikely
        **Confidence: {(100 - risk_percent):.1f}%**
        
        This patient shows low risk of diabetes.
        Maintain a healthy lifestyle.
        """)

    # Risk meter
    st.markdown("#### 📊 Risk Level")
    st.progress(int(risk_percent))
    st.markdown(f"**Diabetes Risk: {risk_percent:.1f}%**")

    # ─────────────────────────────────
    # SHAP Explanation
    # ─────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 Why This Prediction? (SHAP Explanation)")
    st.markdown("""
    The chart below shows which features contributed 
    to this prediction and by how much.
    - 🔴 **Red** = Increases diabetes risk
    - 🔵 **Blue** = Decreases diabetes risk
    """)

    # Calculate SHAP values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
                    pd.DataFrame(input_scaled,
                                 columns=feature_names))

    # SHAP waterfall plot
    explanation = shap.Explanation(
        values        = shap_values[0],
        base_values   = explainer.expected_value,
        data          = input_scaled[0],
        feature_names = feature_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
    plt.close()

    # Simple explanation table
    st.markdown("#### 📋 Feature Contribution Table")

    shap_df = pd.DataFrame({
        'Feature'     : feature_names,
        'Your Value'  : input_data[0],
        'SHAP Impact' : shap_values[0]
    }).sort_values('SHAP Impact', ascending=False)

    shap_df['Direction'] = shap_df['SHAP Impact'].apply(
        lambda x: '🔴 Increases Risk' if x > 0 else '🔵 Decreases Risk')

    shap_df['SHAP Impact'] = shap_df['SHAP Impact'].round(4)

    st.dataframe(
        shap_df[['Feature', 'Your Value',
                 'SHAP Impact', 'Direction']],
        use_container_width = True,
        hide_index          = True)

    # Footer note
    st.markdown("---")
    st.markdown("""
    > ⚠️ **Disclaimer:** This tool is for educational purposes only.
    > Always consult a qualified medical professional for diagnosis.
    """)