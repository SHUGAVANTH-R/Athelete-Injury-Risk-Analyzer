"""Train the genuine Module 2 autoencoder on standardized athlete features."""

import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoencoder import build_autoencoder
from train import load_data


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    x_train, _, _, _ = load_data()
    autoencoder, encoder = build_autoencoder(x_train.shape[1])
    history = autoencoder.fit(x_train, x_train, validation_split=0.20,
                              epochs=50, batch_size=32, verbose=1)
    autoencoder.save("models/feature_autoencoder.keras")
    encoder.save("models/feature_encoder.keras")
    with open("results/autoencoder_history.pkl", "wb") as file:
        pickle.dump(history.history, file)

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training reconstruction loss")
    plt.plot(history.history["val_loss"], label="Validation reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared reconstruction error")
    plt.title("Module 2 Autoencoder Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/autoencoder_loss_curve.png", dpi=150)
    plt.close()
    print("Saved autoencoder models and reconstruction-loss curve.")


if __name__ == "__main__":
    main()
