"""
EDA.py - Exploratory Data Analysis for the Athlete Injury Risk Dataset

Generates all EDA visualizations into results/ for the project report and
checklist Phase 03 compliance (class distribution, feature distributions,
correlation heatmap, missing value check, sample visualizations).

Run:  python notebooks/EDA.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Athlete.xlsx")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
df = pd.read_excel(DATASET_PATH)
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Missing Value Check
# ---------------------------------------------------------------------------
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")

# ---------------------------------------------------------------------------
# 3. Data Types & Basic Statistics
# ---------------------------------------------------------------------------
print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))

# ---------------------------------------------------------------------------
# 4. Class Distribution (Target Variable)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

class_counts = df["Injury_Risk"].value_counts()
labels = ["Low Risk (0)", "High Risk (1)"]
colors = ["#2196F3", "#F44336"]

axes[0].bar(labels, class_counts.values, color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

axes[1].pie(class_counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, explode=(0.02, 0.05))
axes[1].set_title("Class Proportion", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Saved class_distribution.png")

# ---------------------------------------------------------------------------
# 5. Feature Distributions (Histograms)
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("Injury_Risk")
n_cols = 4
n_rows = int(np.ceil(len(numeric_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, color="#42A5F5", edgecolor="black", linewidth=0.3, alpha=0.85)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved feature_distributions.png")

# ---------------------------------------------------------------------------
# 6. Correlation Heatmap
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 11))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 7. Box Plots by Injury Risk (Sample Visualizations)
# ---------------------------------------------------------------------------
key_features = ["Age", "BMI", "Training_Intensity", "Sleep_Hours",
                "Stress_Level", "Recovery_Time", "Flexibility_Score", "Muscle_Asymmetry"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(key_features):
    sns.boxplot(x="Injury_Risk", y=col, data=df, ax=axes[i],
                hue="Injury_Risk", palette={"0": "#2196F3", "1": "#F44336",
                                             0: "#2196F3", 1: "#F44336"},
                legend=False)
    axes[i].set_title(col, fontsize=12, fontweight="bold")
    axes[i].set_xticklabels(["Low Risk", "High Risk"])
    axes[i].set_xlabel("")

plt.suptitle("Feature Box Plots by Injury Risk", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "boxplots_by_risk.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved boxplots_by_risk.png")

# ---------------------------------------------------------------------------
# 8. Pair Plot (Subset of Key Features)
# ---------------------------------------------------------------------------
pair_cols = ["BMI", "Training_Intensity", "Sleep_Hours", "Stress_Level", "Injury_Risk"]
g = sns.pairplot(df[pair_cols], hue="Injury_Risk", palette={0: "#2196F3", 1: "#F44336"},
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
g.figure.suptitle("Pair Plot - Key Features", fontsize=16, fontweight="bold", y=1.02)
g.savefig(os.path.join(RESULTS_DIR, "pairplot_key_features.png"), dpi=150, bbox_inches="tight")
plt.close("all")
print("[OK] Saved pairplot_key_features.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"Total samples: {len(df)}")
print(f"Features: {len(numeric_cols)} numeric + 1 target")
print(f"Class balance: Low Risk = {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%), "
      f"High Risk = {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")
print(f"Missing values: {missing.sum()}")
print(f"\nAll plots saved to {RESULTS_DIR}/")
