"""
train.py

Standalone training script.

CHANGE LOG vs original train.py:
- The original file called model.fit(X_train, y_train, ...) but never
  defined `model`, `X_train`, or `y_train` anywhere in the file, so it
  raised NameError the moment it was run on its own (it only worked
  if pasted into a notebook cell after other cells had already run).
  This version wires preprocess.py and model.py together so the file
  is genuinely standalone, satisfying the checklist's "Code is
  modular" + "Code runs without errors" criteria.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from model import build_model
from preprocess import DataPreprocessor

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_data(dataset_path=None):
    """
    Reuses cached splits from models/ if preprocess.py already ran
    (fast path for re-training), otherwise runs preprocessing fresh
    from the given Excel file.
    """
    cached = all(
        os.path.exists(os.path.join("models", f))
        for f in ["X_train.npy", "X_test.npy", "y_train.pkl", "y_test.pkl"]
    )

    if cached:
        X_train = np.load("models/X_train.npy")
        X_test = np.load("models/X_test.npy")
        y_train = pd.read_pickle("models/y_train.pkl")
        y_test = pd.read_pickle("models/y_test.pkl")
        return X_train, X_test, y_train, y_test

    if dataset_path is None:
        raise FileNotFoundError(
            "No cached splits found in models/. Run "
            "`python preprocess.py <path_to_Athlete.xlsx>` first, or "
            "pass the dataset path: `python train.py <path_to_Athlete.xlsx>`."
        )

    processor = DataPreprocessor(dataset_path)
    return processor.process()


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    X_train, X_test, y_train, y_test = load_data(dataset_path)

    model = build_model(X_train.shape[1])
    model.summary()

    checkpoint = ModelCheckpoint(
        filepath="models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    print("\nTraining Completed Successfully!")

    with open("results/history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    print("History Saved Successfully!")


if __name__ == "__main__":
    main()
