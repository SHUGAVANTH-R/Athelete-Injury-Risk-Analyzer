import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from explainability import (
    load_explainability_assets,
    build_shap_explainer,
    build_lime_explainer,
    explain_with_shap,
    explain_with_lime,
)

model = tf.keras.models.load_model("models/best_model.keras")
scaler = joblib.load("models/scaler.pkl")
feature_names, background = load_explainability_assets()


@st.cache_resource
def get_explainers():
    shap_explainer = build_shap_explainer(model, background)
    lime_explainer = build_lime_explainer(background, feature_names)
    return shap_explainer, lime_explainer


shap_explainer, lime_explainer = get_explainers()

st.set_page_config(
    page_title="Athlete Injury Risk Analyzer",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Athlete Injury Risk Analyzer")
st.write("Predict athlete injury risk using Deep Learning.")
st.divider()


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 15, 60, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 120, 230, 175)
    weight = st.number_input("Weight (kg)", 30, 150, 70)
    bmi = st.number_input("BMI", 10.0, 40.0, 23.5)
    training_frequency = st.slider("Training Frequency", 1, 7, 5)
    training_duration = st.slider("Training Duration (min)", 30, 240, 90)

with col2:
    warmup = st.slider("Warm-up Time", 0, 30, 10)
    sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
    flexibility = st.slider("Flexibility Score", 1, 10, 6)
    muscle = st.slider("Muscle Asymmetry", 1, 10, 4)
    recovery = st.slider("Recovery Time", 1, 72, 24)
    # NOTE: the training data (Athlete.xlsx) has Injury_History as 4
    # distinct levels (0-3), not a strict Yes/No. This selectbox is a
    # simplification — see the README note on this field. If you have
    # the true count of previous injuries, prefer a number_input(0,3).
    injury = st.selectbox("Previous Injury", ["No", "Yes"])
    stress = st.slider("Stress Level", 1, 10, 4)
    intensity = st.slider("Training Intensity", 1, 10, 5)


if st.button("Predict Injury Risk"):

    gender_val = 1 if gender == "Male" else 0
    injury_val = 1 if injury == "Yes" else 0

    raw_features = [
        age, gender_val, height, weight, bmi,
        training_frequency, training_duration, warmup, sleep,
        flexibility, muscle, recovery, injury_val, stress, intensity
    ]

    features = np.array([raw_features])
    features_scaled = scaler.transform(features)

    probability = model.predict(features_scaled, verbose=0)[0][0]
    confidence = probability * 100

    if probability >= 0.70:
        risk = "🔴 HIGH RISK"
    elif probability >= 0.40:
        risk = "🟠 MODERATE RISK"
    else:
        risk = "🟢 LOW RISK"

    st.subheader("Prediction")
    st.success(risk)
    st.metric("Confidence", f"{confidence:.2f}%")
    st.progress(float(probability))

    st.divider()

    st.subheader("Recommendations")

    recommendations = []
    if sleep < 6:
        recommendations.append("😴 Increase sleep to 7–9 hours.")
    if stress >= 7:
        recommendations.append("🧘 Reduce stress through meditation or breathing exercises.")
    if intensity >= 8:
        recommendations.append("🏋 Reduce training intensity by 15–20%.")
    if recovery < 24:
        recommendations.append("🛌 Increase recovery time between sessions.")
    if flexibility < 5:
        recommendations.append("🤸 Improve flexibility with daily stretching.")
    if bmi > 27:
        recommendations.append("🥗 Maintain a healthy BMI through nutrition and exercise.")
    if injury_val == 1:
        recommendations.append("🩺 Consult a physiotherapist before high-intensity training.")
    if training_frequency > 6:
        recommendations.append("📅 Include at least one complete rest day each week.")
    if not recommendations:
        recommendations.append("✅ Excellent! Continue your current routine.")

    for item in recommendations:
        st.write(item)

    st.divider()

    # ---------------- Explainable AI: SHAP + LIME ----------------
    st.subheader("🔍 Why this prediction? (Explainable AI)")

    tab_shap, tab_lime = st.tabs(["SHAP", "LIME"])

    with tab_shap:
        with st.spinner("Computing SHAP feature contributions..."):
            shap_values, base_value = explain_with_shap(shap_explainer, features_scaled)

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values
        }).sort_values("Impact", key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_df["Impact"]]
        ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
        ax.set_xlabel("Impact on predicted injury risk")
        ax.set_title("SHAP feature contributions (this athlete)")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)
        st.caption(
            "🔴 Red bars push the prediction toward HIGH risk. "
            "🔵 Blue bars push it toward LOW risk. "
            f"Baseline (average) predicted risk: {base_value:.2%}"
        )

    with tab_lime:
        with st.spinner("Computing LIME explanation..."):
            lime_result = explain_with_lime(lime_explainer, model, features_scaled[0])

        lime_df = pd.DataFrame(lime_result, columns=["Condition", "Weight"])
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        colors2 = ["#d62728" if v > 0 else "#1f77b4" for v in lime_df["Weight"]]
        ax2.barh(lime_df["Condition"], lime_df["Weight"], color=colors2)
        ax2.set_xlabel("Weight toward HIGH risk")
        ax2.set_title("LIME local explanation (this athlete)")
        ax2.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig2)
        st.caption("Each bar is a local rule LIME found for this specific athlete's inputs.")
