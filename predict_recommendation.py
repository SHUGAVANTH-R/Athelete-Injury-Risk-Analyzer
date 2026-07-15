import joblib
import numpy as np
import tensorflow as tf

from explainability import (
    load_explainability_assets,
    build_shap_explainer,
    build_lime_explainer,
    explain_with_shap,
    explain_with_lime,
)

# Load trained model
model = tf.keras.models.load_model("models/best_model.keras")

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Explainability assets (lazy-built on first use, since KernelExplainer
# setup has a small fixed cost)
_feature_names, _background = load_explainability_assets()
_shap_explainer = None
_lime_explainer = None


def predict_risk(features):
    """
    features should be a list in this order:

    Age, Gender, Height_cm, Weight_kg, BMI, Training_Frequency,
    Training_Duration, Warmup_Time, Sleep_Hours, Flexibility_Score,
    Muscle_Asymmetry, Recovery_Time, Injury_History, Stress_Level,
    Training_Intensity
    """
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    probability = model.predict(features, verbose=0)[0][0]
    confidence = probability * 100

    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return risk, confidence


def explain_prediction(features, method="shap", num_features=8):
    """
    Returns a feature-attribution explanation for one athlete's raw
    (unscaled) feature list, using either SHAP or LIME.

    method: "shap" or "lime"
    """
    global _shap_explainer, _lime_explainer

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    if method == "shap":
        if _shap_explainer is None:
            _shap_explainer = build_shap_explainer(model, _background)
        shap_values, base_value = explain_with_shap(_shap_explainer, features_scaled)
        return sorted(
            zip(_feature_names, shap_values),
            key=lambda pair: -abs(pair[1])
        )[:num_features]

    if method == "lime":
        if _lime_explainer is None:
            _lime_explainer = build_lime_explainer(_background, _feature_names)
        return explain_with_lime(_lime_explainer, model, features_scaled[0], num_features)

    raise ValueError("method must be 'shap' or 'lime'")


def generate_recommendations(data):

    recommendations = []

    if data["Sleep_Hours"] < 6:
        recommendations.append("Increase sleep to 7-9 hours every night.")

    if data["Stress_Level"] >= 7:
        recommendations.append("Practice meditation, yoga, or breathing exercises to reduce stress.")

    if data["Training_Intensity"] >= 8:
        recommendations.append("Reduce training intensity by 15-20% for the next few sessions.")

    if data["Recovery_Time"] < 24:
        recommendations.append("Increase recovery time between training sessions.")

    if data["Flexibility_Score"] < 5:
        recommendations.append("Include stretching and mobility exercises daily.")

    if data["Muscle_Asymmetry"] > 6:
        recommendations.append("Perform unilateral strength exercises to reduce muscle imbalance.")

    if data["BMI"] > 27:
        recommendations.append("Follow a balanced nutrition plan to maintain a healthy BMI.")

    if data["Warmup_Time"] < 10:
        recommendations.append("Increase warm-up duration to at least 10-15 minutes.")

    if data["Injury_History"] == 1:
        recommendations.append("Consult a physiotherapist before high-intensity workouts.")

    if data["Training_Frequency"] > 6:
        recommendations.append("Consider one complete recovery day each week.")

    if data["Age"] > 35:
        recommendations.append("Focus on recovery, mobility, and strength maintenance.")

    if len(recommendations) == 0:
        recommendations.append(
            "Great! Your current training routine appears balanced. Maintain your healthy habits."
        )

    return recommendations
