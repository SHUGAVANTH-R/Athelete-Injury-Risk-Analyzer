"""Module 2: autoencoder for unsupervised athlete-feature representation learning."""

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_shape: int, latent_dim: int = 16):
    """Build a symmetric feature-reconstruction autoencoder and its encoder."""
    inputs = Input(shape=(input_shape,), name="athlete_features")
    encoded = Dense(64, activation="relu", name="encoder_dense_64")(inputs)
    encoded = Dense(32, activation="relu", name="encoder_dense_32")(encoded)
    latent = Dense(latent_dim, activation="relu", name="latent_representation")(encoded)
    decoded = Dense(32, activation="relu", name="decoder_dense_32")(latent)
    decoded = Dense(64, activation="relu", name="decoder_dense_64")(decoded)
    reconstruction = Dense(input_shape, activation="linear", name="reconstruction")(decoded)
    autoencoder = Model(inputs, reconstruction, name="Athlete_Feature_Autoencoder")
    encoder = Model(inputs, latent, name="Athlete_Feature_Encoder")
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return autoencoder, encoder


if __name__ == "__main__":
    model, encoder_model = build_autoencoder(15)
    model.summary()
    encoder_model.summary()
