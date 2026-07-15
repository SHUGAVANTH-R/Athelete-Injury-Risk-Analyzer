# -*- coding: utf-8 -*-
"""
preprocess.py

Loads the athlete dataset, cleans it, encodes categorical columns,
splits it, scales it, and persists every artifact (scaler, encoders,
feature names, and a background sample) that later stages
(train.py, evaluate.py, explainability.py, app.py) depend on.

CHANGE LOG vs original preprocess.py:
- Saves models/feature_names.pkl (needed by SHAP/LIME/app.py so the
  15 columns are always used in the exact order the model expects).
- Saves models/background_data.pkl: a small scaled sample of the
  training set, used as the SHAP/LIME reference distribution so the
  Streamlit app does not need to reload the full training set.
- The __main__ block no longer hard-imports google.colab. It only
  uses the Colab uploader when it detects it is actually running
  inside Colab, otherwise it accepts a normal file path argument so
  the script also runs locally / in CI, which the checklist's
  "Code runs without errors (verified during demo)" item requires.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class DataPreprocessor:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.scaler = StandardScaler()
        self.gender_encoder = LabelEncoder()
        self.injury_encoder = LabelEncoder()

    def load_dataset(self):
        self.df = pd.read_excel(self.dataset_path)

        print("=" * 60)
        print("Dataset Loaded Successfully")
        print("=" * 60)
        print(self.df.head())

        return self.df

    def clean_data(self):
        print("\nRemoving duplicate rows...")
        self.df.drop_duplicates(inplace=True)

        print("Checking missing values...")
        print(self.df.isnull().sum())

        numerical = self.df.select_dtypes(include=["int64", "float64"]).columns
        for col in numerical:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        categorical = self.df.select_dtypes(include=["object"]).columns
        for col in categorical:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    def encode_features(self):
        # NOTE: in the current Athlete.xlsx, Gender and Injury_History are
        # already numeric. LabelEncoder is kept here so the pipeline still
        # works if a future raw export uses string labels ("Male"/"Female",
        # "Yes"/"No"), but be aware Injury_History in the current dataset
        # actually has 4 distinct levels (0-3), not a binary Yes/No — see
        # the audit notes for why this matters for app.py.
        self.df["Gender"] = self.gender_encoder.fit_transform(self.df["Gender"])
        self.df["Injury_History"] = self.injury_encoder.fit_transform(self.df["Injury_History"])

        joblib.dump(self.gender_encoder, os.path.join(MODEL_DIR, "gender_encoder.pkl"))
        joblib.dump(self.injury_encoder, os.path.join(MODEL_DIR, "injury_encoder.pkl"))

        return self.df

    def split_dataset(self):
        X = self.df.drop("Injury_Risk", axis=1)
        y = self.df["Injury_Risk"]

        joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))

        return train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        # Background sample for SHAP/LIME: 100 rows (or fewer if the
        # training set is smaller) drawn from the *scaled* training data,
        # since that is exactly the space the model and the Streamlit
        # app operate in.
        rng = np.random.default_rng(42)
        n_bg = min(100, X_train_scaled.shape[0])
        idx = rng.choice(X_train_scaled.shape[0], size=n_bg, replace=False)
        background = X_train_scaled[idx]
        joblib.dump(background, os.path.join(MODEL_DIR, "background_data.pkl"))

        return X_train_scaled, X_test_scaled

    def process(self):
        self.load_dataset()
        self.clean_data()
        self.encode_features()

        X_train, X_test, y_train, y_test = self.split_dataset()
        X_train, X_test = self.scale_features(X_train, X_test)

        print("\nPreprocessing Completed Successfully")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    # Runs both inside Google Colab (keeps the original upload widget)
    # and as a normal local script: `python preprocess.py Athlete.xlsx`
    try:
        import google.colab  # noqa: F401
        from google.colab import files
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
    except ImportError:
        if len(sys.argv) < 2:
            print("Usage: python preprocess.py <path_to_Athlete.xlsx>")
            sys.exit(1)
        filename = sys.argv[1]

    processor = DataPreprocessor(filename)
    X_train, X_test, y_train, y_test = processor.process()

    # Persist splits so train.py / evaluate.py can be run as separate
    # standalone scripts without re-running preprocessing each time.
    np.save(os.path.join(MODEL_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
    y_train.to_pickle(os.path.join(MODEL_DIR, "y_train.pkl"))
    y_test.to_pickle(os.path.join(MODEL_DIR, "y_test.pkl"))
    print("\nSaved train/test splits to models/ for train.py and evaluate.py")
