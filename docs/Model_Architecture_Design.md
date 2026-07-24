# Model Architecture Design — Athlete Injury Risk Analyzer

## Architecture Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
│                  15 Features                                    │
│  [Age, Gender, Height, Weight, BMI, Training_Freq, Duration,   │
│   Warmup, Sleep, Flexibility, Muscle_Asymmetry, Recovery,      │
│   Injury_History, Stress, Training_Intensity]                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 1 — Dense(128, ReLU)                 │
│                  BatchNormalization()                           │
│                  Dropout(0.30)                                  │
│  Purpose: Learn broad, high-level feature interactions         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 2 — Dense(64, ReLU)                  │
│                  BatchNormalization()                           │
│                  Dropout(0.30)                                  │
│  Purpose: Extract intermediate-level feature combinations      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 3 — Dense(32, ReLU)                  │
│  Purpose: Compress representations for final decision          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              HIDDEN LAYER 4 — Dense(16, ReLU)                  │
│  Purpose: Final feature refinement before classification       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT LAYER — Dense(1, Sigmoid)                  │
│  Output: P(Injury_Risk = High) ∈ [0, 1]                       │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              EXPLAINABILITY LAYER                               │
│         ┌──────────────┬──────────────┐                        │
│         │    SHAP       │    LIME      │                        │
│         │ KernelExpl.   │ TabularExpl. │                        │
│         └──────────────┴──────────────┘                        │
│  Purpose: Post-hoc feature attribution for each prediction     │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              STREAMLIT WEB APP                                  │
│  Interactive input → Prediction → Recommendations + XAI        │
└─────────────────────────────────────────────────────────────────┘
```

> Note: A graphical version of this diagram is saved as `docs/architecture.png`.

---

## Layer-by-Layer Explanation

### Input Layer
- **Shape:** (batch_size, 15)
- **Description:** Accepts 15 standardized numeric features representing the athlete's physical attributes, training load, lifestyle factors, and injury history. All features are pre-scaled using `StandardScaler` (zero mean, unit variance).

### Hidden Layer 1 — Dense(128, ReLU) + BatchNorm + Dropout(0.30)
- **Units:** 128 neurons with ReLU activation
- **BatchNormalization:** Normalizes activations across the mini-batch, stabilizing gradients and accelerating convergence. This is critical for our relatively small dataset (1000 samples) where training can be noisy.
- **Dropout(0.30):** Randomly deactivates 30% of neurons during training to prevent co-adaptation and overfitting. The dropout rate of 0.30 was chosen as a balance between regularization strength and information preservation.
- **Purpose:** This widest layer captures broad, high-dimensional feature interactions — for example, learning that high BMI + low flexibility + high training intensity together create a compound risk.

### Hidden Layer 2 — Dense(64, ReLU) + BatchNorm + Dropout(0.30)
- **Units:** 64 neurons with ReLU activation
- **BatchNormalization + Dropout(0.30):** Same regularization strategy as Layer 1.
- **Purpose:** Compresses the 128-dimensional representation to 64 dimensions, forcing the network to prioritize the most informative feature combinations. The 2:1 compression ratio is a standard practice in funnel-shaped MLP architectures.

### Hidden Layer 3 — Dense(32, ReLU)
- **Units:** 32 neurons with ReLU activation
- **No BatchNorm/Dropout:** The deeper layers have fewer parameters and are less prone to overfitting. Removing regularization here allows the model to preserve learned representations more faithfully.
- **Purpose:** Further compresses representations, learning the core risk indicators.

### Hidden Layer 4 — Dense(16, ReLU)
- **Units:** 16 neurons with ReLU activation
- **Purpose:** Final feature refinement before the output decision. This layer produces a compact 16-dimensional encoding that captures the essence of injury risk.

### Output Layer — Dense(1, Sigmoid)
- **Units:** 1 neuron with Sigmoid activation
- **Output:** A single probability P(High Risk) ∈ [0, 1]
- **Decision Thresholds:** ≥ 0.70 → HIGH RISK, 0.40–0.69 → MODERATE RISK, < 0.40 → LOW RISK

### Training Configuration

| Parameter | Value | Justification |
|---|---|---|
| Optimizer | Adam (lr=0.001) | Adaptive learning rate; efficient for sparse gradients |
| Loss Function | Binary Cross-Entropy | Standard for binary classification |
| Batch Size | 32 | Balance between gradient noise and computational efficiency |
| Max Epochs | 100 | Upper bound; EarlyStopping usually terminates earlier |
| EarlyStopping | patience=10, restore_best_weights | Prevents overfitting; reverts to best checkpoint |
| ReduceLROnPlateau | factor=0.5, patience=5, min_lr=1e-6 | Halves LR when validation loss plateaus |
| ModelCheckpoint | save_best_only, monitor=val_accuracy | Saves only the model with highest validation accuracy |
| Validation Split | 20% of training data | Used for monitoring; test set is held out completely |

---

## Module Mapping

| Module | Concept Used | Where Applied |
|---|---|---|
| **Module 1** (MLP / RNN / LSTM / Hopfield / Boltzmann) | **Multi-Layer Perceptron (MLP)** | `model.py` — 4 hidden layers with ReLU, BatchNorm, Dropout |
| **Module 2** (CNN / Autoencoder / GAN / Attention / Transfer Learning / DRL) | **Autoencoder-inspired progressive compression** | The funnel architecture (128→64→32→16) mirrors an autoencoder's encoder, progressively compressing input features into a compact risk representation |
| **Module 3** (Real-World Application) | **Explainable AI (SHAP + LIME) + Streamlit Deployment** | `explainability.py` + `app.py` — real-time predictions with feature-attribution explanations |

---

## Justification for Architectural Choices

1. **MLP over CNN/RNN:** The input is fixed-length tabular data (15 features), not images or sequences. MLP is the natural fit; CNN would add unnecessary spatial assumptions and RNN would add unnecessary temporal structure.

2. **Funnel shape (128→64→32→16):** Progressive dimensionality reduction forces the network to learn increasingly abstract representations. This is inspired by autoencoder encoder architectures and is a well-established pattern for tabular classification.

3. **BatchNormalization in early layers:** With only 1000 training samples, internal covariate shift can destabilize training. BatchNorm mitigates this and allows higher learning rates.

4. **Dropout only in early layers:** The first two layers have the most parameters (128×15 + 64×128 = 10,112) and are most prone to overfitting. The last two layers have far fewer parameters (32×64 + 16×32 = 2,560) and need less regularization.

5. **Sigmoid output:** Binary classification with a single output neuron is more parameter-efficient than softmax with 2 outputs, and directly produces a probability for thresholding.

6. **SHAP + LIME for explainability:** Medical/sports-science applications demand interpretability. SHAP provides theoretically grounded Shapley values, while LIME provides intuitive local linear approximations — together they give comprehensive explanations.
