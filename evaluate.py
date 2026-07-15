"""
evaluate.py

Standalone evaluation script.

CHANGE LOG vs original evaluate.py:
- The original referenced `model`, `X_test`, `y_test`, and `history`
  without defining or loading any of them anywhere in the file. This
  version loads the saved model (models/best_model.keras), the saved
  test split (models/X_test.npy, models/y_test.pkl), and the saved
  training history (results/history.pkl) so the script actually runs
  on its own.
- plt.show() is replaced with plt.savefig(...) into results/, since
  plt.show() blocks / no-ops in headless environments (CI, servers,
  the grading environment) — the figures are still produced, just as
  files instead of pop-up windows.
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

os.makedirs("results", exist_ok=True)

model = tf.keras.models.load_model("models/best_model.keras")
X_test = np.load("models/X_test.npy")
y_test = pd.read_pickle("models/y_test.pkl")

y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC AUC Score : {roc_auc:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("results/roc_curve.png", bbox_inches="tight")
plt.close()

# Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)
plt.figure(figsize=(6, 5))
plt.plot(recall_curve, precision_curve, color="green", label=f"AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.legend()
plt.grid()
plt.savefig("results/precision_recall_curve.png", bbox_inches="tight")
plt.close()

# Training curves — only if history.pkl exists (i.e. train.py has been run)
history_path = "results/history.pkl"
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(7, 5))
    plt.plot(history["accuracy"], linewidth=2, label="Training Accuracy")
    plt.plot(history["val_accuracy"], linewidth=2, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("results/accuracy_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["loss"], linewidth=2, label="Training Loss")
    plt.plot(history["val_loss"], linewidth=2, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("results/loss_curve.png", bbox_inches="tight")
    plt.close()
else:
    print("\n(results/history.pkl not found — skipping accuracy/loss curves. "
          "Run train.py first to generate training history.)")

print("\nAll evaluation plots saved to results/")
print("Evaluation Completed Successfully!")
