"""
explainability.py

Adds the Explainable AI layer that was documented (Model_Architecture_
Design.txt, PROPOSAL.md, LITERATURE_SURVEY.md all mention SHAP) but was
never actually implemented in any .py file. This module provides both
SHAP and LIME explanations for a single athlete's prediction, and is
built to be imported by app.py, predict_recommendation.py, or a
notebook.

Design notes:
- Uses shap.KernelExplainer(model.predict, background) rather than
  shap.DeepExplainer. KernelExplainer is model-agnostic (works
  regardless of Keras/TF version quirks) and is fast enough here
  because the background sample is small (<=100 rows) and we only
  explain one instance at a time in the app.
- The background sample is the one persisted by preprocess.py
  (models/background_data.pkl) so the app does not need to reload
  the full training set at runtime.
- LIME uses LimeTabularExplainer in "classification" mode. Its
  predict_fn must return probabilities for BOTH classes (shape
  (n, 2)), unlike the model itself which only outputs P(High Risk).
"""

import os
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

MODEL_DIR = "models"


def load_explainability_assets():
    """Loads the feature names and background sample saved by preprocess.py."""
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    background = joblib.load(os.path.join(MODEL_DIR, "background_data.pkl"))
    return feature_names, background


def build_shap_explainer(model, background):
    """
    model: a loaded tf.keras model whose .predict returns P(High Risk)
           with shape (n, 1).
    background: scaled reference sample, shape (n_bg, n_features).
    """
    def predict_fn(x):
        return model.predict(x, verbose=0).ravel()

    return shap.KernelExplainer(predict_fn, background)


def build_lime_explainer(background, feature_names):
    return LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=["Low Risk", "High Risk"],
        mode="classification",
        discretize_continuous=True
    )


def explain_with_shap(explainer, scaled_instance, nsamples=100):
    """
    scaled_instance: shape (1, n_features), already scaled.
    Returns a list of (feature_index_value_pairs) sorted by |impact|,
    plus the model's baseline (expected) output.
    """
    shap_values = explainer.shap_values(scaled_instance, nsamples=nsamples)
    shap_values = np.array(shap_values).reshape(-1)
    return shap_values, explainer.expected_value


def explain_with_lime(explainer, model, scaled_instance, num_features=8):
    """
    scaled_instance: shape (n_features,) 1D, already scaled.
    """
    def predict_proba(x):
        p1 = model.predict(x, verbose=0).ravel()
        return np.column_stack([1 - p1, p1])

    exp = explainer.explain_instance(
        scaled_instance,
        predict_proba,
        num_features=num_features
    )
    return exp.as_list()


if __name__ == "__main__":
    # Quick standalone smoke test:
    # python explainability.py
    import tensorflow as tf

    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))
    feature_names, background = load_explainability_assets()

    instance = background[0:1]  # just testing plumbing, not a real prediction

    shap_explainer = build_shap_explainer(model, background)
    shap_values, base_value = explain_with_shap(shap_explainer, instance)
    print("SHAP base value:", base_value)
    for name, val in sorted(zip(feature_names, shap_values), key=lambda x: -abs(x[1])):
        print(f"  {name:20s} {val:+.4f}")

    lime_explainer = build_lime_explainer(background, feature_names)
    lime_result = explain_with_lime(lime_explainer, model, instance[0])
    print("\nLIME:")
    for feat, weight in lime_result:
        print(f"  {feat:30s} {weight:+.4f}")
