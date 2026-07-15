# 🏃 Athlete Injury Risk Analyzer

A Streamlit app that predicts an athlete's injury risk using a deep learning
model (Keras/TensorFlow), with SHAP and LIME explainability so you can see
*why* a given prediction was made.

## Project structure

| File | Purpose |
|---|---|
| `preprocess.py` | Loads `Athlete.xlsx`, cleans it, encodes/splits/scales it, saves artifacts to `models/` |
| `model.py` | Defines the Keras neural network architecture |
| `train.py` | Trains the model, saves `models/best_model.keras` |
| `evaluate.py` | Evaluates the trained model, saves plots to `results/` |
| `explainability.py` | SHAP + LIME explainers used by the app |
| `predict_recommendation.py` | Standalone CLI-style prediction + recommendation logic |
| `app.py` | The Streamlit app (UI) |
| `requirements.txt` | Python dependencies |

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Prepare the data & train the model

Place `Athlete.xlsx` in the project root, then run, in order:

```bash
python preprocess.py Athlete.xlsx   # generates models/scaler.pkl, encoders, feature_names, splits
python train.py                     # trains the network, saves models/best_model.keras
python evaluate.py                  # generates results/ plots (confusion matrix, ROC, etc.)
```

## 3. Run the app locally

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.



## Notes

- `Injury_History` in the training data has more granularity (0–3) than the
  simple Yes/No toggle in the current `app.py` UI — see the comment in
  `app.py` if you want to change this to a numeric input.
