# Project Proposal — Deep Learning-Based Athlete Injury Risk Analyzer

| Field | Details |
|---|---|
| **Student Name** | SHUGAVANTH R |
| **Roll Number** | 727824TUAM046 |
| **Section / Year** | III – CSE (AIML) |
| **Institution** | Sri Krishna College of Technology, Coimbatore |
| **Course Code** | 23ADC04 — Deep Learning |
| **Project Title** | Deep Learning-Based Athlete Injury Risk Analyzer |
| **Date** | July 2026 |

---

## 1. Problem Statement

Sports injuries significantly affect athletes' careers and well-being. Traditional injury risk assessment relies on subjective evaluation by physiotherapists and coaches, which is inconsistent, time-consuming, and unavailable to many athletes. There is a need for an **automated, data-driven system** that can predict injury risk from measurable physiological and training parameters.

## 2. Objective

Design and implement a **deep neural network** that:

1. Ingests athlete biometric, training, and lifestyle data (15 features).
2. Predicts whether the athlete is at **high risk** of injury (binary classification).
3. Provides **explainable AI (XAI)** insights via SHAP and LIME so that coaches and athletes understand *which factors* contribute most to the risk.
4. Delivers predictions through an interactive **Streamlit web application** for real-time use.

**Expected Outcome:** A deployable web app achieving ≥ 85 % accuracy on the test set, with per-prediction SHAP/LIME explanations.

## 3. Dataset Source

| Property | Value |
|---|---|
| Name | Athlete Injury Risk Dataset |
| Format | Excel (.xlsx) |
| Samples | 1 000 |
| Features | 15 numeric / encoded features + 1 binary target (`Injury_Risk`) |
| Source | Curated sports-science dataset; features align with established injury-risk factors from the literature (see Literature Survey) |

### Feature List

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | Age | int | Athlete's age (18–40) |
| 2 | Gender | int (encoded) | 0 = Female, 1 = Male |
| 3 | Height_cm | float | Height in centimeters |
| 4 | Weight_kg | float | Weight in kilograms |
| 5 | BMI | float | Body Mass Index |
| 6 | Training_Frequency | int | Sessions per week |
| 7 | Training_Duration | int | Minutes per session |
| 8 | Warmup_Time | int | Warm-up minutes |
| 9 | Sleep_Hours | float | Average daily sleep |
| 10 | Flexibility_Score | float | Self-reported 1–10 |
| 11 | Muscle_Asymmetry | float | Asymmetry index 1–10 |
| 12 | Recovery_Time | int | Hours between sessions |
| 13 | Injury_History | int | Past injuries (0–3) |
| 14 | Stress_Level | int | Perceived stress 1–10 |
| 15 | Training_Intensity | float | Self-reported 1–10 |
| **Target** | Injury_Risk | int | 0 = Low Risk, 1 = High Risk |

## 4. Proposed Architecture / Methodology

### 4.1 Pipeline Overview

```
Raw Data → Preprocessing → Train/Test Split → DNN Training → Evaluation → Explainability → Streamlit App
```

### 4.2 Deep Neural Network Architecture

A **Multi-Layer Perceptron (MLP)** — the core Module 1 concept — with the following design:

| Layer | Units | Activation | Regularization |
|---|---|---|---|
| Input | 15 | — | — |
| Dense + BatchNorm + Dropout | 128 | ReLU | Dropout 0.30 |
| Dense + BatchNorm + Dropout | 64 | ReLU | Dropout 0.30 |
| Dense | 32 | ReLU | — |
| Dense | 16 | ReLU | — |
| Output | 1 | Sigmoid | — |

- **Optimizer:** Adam (lr = 0.001, with ReduceLROnPlateau)
- **Loss:** Binary Cross-Entropy
- **Callbacks:** EarlyStopping (patience 10), ModelCheckpoint, ReduceLROnPlateau

### 4.3 Explainability Layer (Module 3 — Real-World Application)

- **SHAP (KernelExplainer):** Model-agnostic feature attribution for individual predictions.
- **LIME (LimeTabularExplainer):** Local interpretable model-agnostic explanations.

### 4.4 Deployment

- **Streamlit** web application with interactive sliders for all 15 features, real-time prediction, and embedded SHAP/LIME visualizations.

## 5. Tools & Technologies

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| DL Framework | TensorFlow / Keras |
| ML Utilities | scikit-learn |
| Explainability | SHAP, LIME |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git + GitHub |

## 6. Timeline

| Week | Activity |
|---|---|
| 1 | Proposal, literature survey, dataset collection |
| 2 | EDA, preprocessing pipeline |
| 3 | Model architecture design & training |
| 4 | Evaluation, explainability integration |
| 5 | Streamlit app, documentation, report |
| 6 | Final demo, viva preparation |

---

*Submitted for approval — 23ADC04 Deep Learning Individual Project.*
