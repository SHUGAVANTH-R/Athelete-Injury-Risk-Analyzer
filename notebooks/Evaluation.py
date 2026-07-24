"""
Evaluation.py - Comprehensive Model Evaluation Notebook (Script Form)

Loads the trained model and test data, computes all evaluation metrics,
generates plots, and compares results against a baseline (Logistic Regression).

This fulfills Phase 06 of the project checklist:
 - Training & validation loss/accuracy curves
 - Confusion matrix
 - Evaluation metrics (accuracy, F1, precision, recall, AUC)
 - Baseline comparison

Run:  python notebooks/Evaluation.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load Model & Data
# ---------------------------------------------------------------------------
print("=" * 60)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 60)

model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_model.keras"))
X_test = np.load(os.path.join(MODELS_DIR, "X_test.npy"))
y_test = pd.read_pickle(os.path.join(MODELS_DIR, "y_test.pkl"))
X_train = np.load(os.path.join(MODELS_DIR, "X_train.npy"))
y_train = pd.read_pickle(os.path.join(MODELS_DIR, "y_train.pkl"))

print(f"Test set: {X_test.shape[0]} samples")
print(f"Train set: {X_train.shape[0]} samples")

# ---------------------------------------------------------------------------
# 1. DNN Predictions
# ---------------------------------------------------------------------------
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n--- DNN Model Performance ---")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")
print(f"ROC AUC Score : {roc_auc:.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# ---------------------------------------------------------------------------
# 2. Baseline Comparison (Logistic Regression)
# ---------------------------------------------------------------------------
print("--- Baseline: Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print(f"Accuracy      : {lr_accuracy:.4f}")
print(f"Precision     : {lr_precision:.4f}")
print(f"Recall        : {lr_recall:.4f}")
print(f"F1 Score      : {lr_f1:.4f}")
print(f"ROC AUC Score : {lr_auc:.4f}")

# ---------------------------------------------------------------------------
# 3. Comparison Table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON: DNN vs Logistic Regression Baseline")
print("=" * 60)
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    "DNN (Ours)": [accuracy, precision, recall, f1, roc_auc],
    "Logistic Regression": [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc],
})
comparison["Improvement"] = comparison["DNN (Ours)"] - comparison["Logistic Regression"]
print(comparison.to_string(index=False))

# Save comparison table
comparison.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"), index=False)
print("\n[OK] Saved baseline_comparison.csv")

# ---------------------------------------------------------------------------
# 4. Confusion Matrix (DNN)
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
plt.title("Confusion Matrix - DNN Model", fontsize=14, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved confusion_matrix.png")

# ---------------------------------------------------------------------------
# 5. ROC Curve (DNN vs Baseline)
# ---------------------------------------------------------------------------
fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr_dnn, tpr_dnn, color="blue", linewidth=2,
         label=f"DNN (AUC = {roc_auc:.3f})")
plt.plot(fpr_lr, tpr_lr, color="green", linewidth=2, linestyle="--",
         label=f"Logistic Regression (AUC = {lr_auc:.3f})")
plt.plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.5)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve - DNN vs Baseline", fontsize=14, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved roc_curve.png")

# ---------------------------------------------------------------------------
# 6. Precision-Recall Curve
# ---------------------------------------------------------------------------
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc_val = auc(rec_curve, prec_curve)

plt.figure(figsize=(7, 6))
plt.plot(rec_curve, prec_curve, color="green", linewidth=2,
         label=f"DNN (PR AUC = {pr_auc_val:.3f})")
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved precision_recall_curve.png")

# ---------------------------------------------------------------------------
# 7. Training Curves (if history available)
# ---------------------------------------------------------------------------
history_path = os.path.join(RESULTS_DIR, "history.pkl")
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["accuracy"], linewidth=2, label="Training Accuracy")
    axes[0].plot(history["val_accuracy"], linewidth=2, label="Validation Accuracy")
    axes[0].set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["loss"], linewidth=2, label="Training Loss", color="red")
    axes[1].plot(history["val_loss"], linewidth=2, label="Validation Loss", color="orange")
    axes[1].set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved training_curves.png")
else:
    print("[!] history.pkl not found - skipping training curves (run train.py first)")

# ---------------------------------------------------------------------------
# 8. Comparison Bar Chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison["Metric"]))
width = 0.35
bars1 = ax.bar(x - width/2, comparison["DNN (Ours)"], width, label="DNN (Ours)",
               color="#1565C0", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, comparison["Logistic Regression"], width,
               label="Logistic Regression (Baseline)", color="#FF8F00",
               edgecolor="black", linewidth=0.5)

ax.set_ylabel("Score", fontsize=12)
ax.set_title("DNN vs Logistic Regression - Metric Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(comparison["Metric"], fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "baseline_comparison_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved baseline_comparison_chart.png")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE - All results saved to results/")
print("=" * 60)
