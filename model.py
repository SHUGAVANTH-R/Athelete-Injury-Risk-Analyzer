import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input
)
from tensorflow.keras.optimizers import Adam


def build_model(input_shape):

    model = Sequential(name="Athlete_Injury_Risk_Model")

    # Input Layer
    model.add(Input(shape=(input_shape,)))

    # Hidden Layer 1
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 2
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    # Hidden Layer 3
    model.add(Dense(32, activation="relu"))

    # Hidden Layer 4
    model.add(Dense(16, activation="relu"))

    # Output Layer
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


if __name__ == "__main__":
    # Quick sanity check: build a model for the 15-feature athlete
    # dataset and print its summary. This is what used to run
    # unconditionally at import time (crashing any script that
    # imported build_model) — now it only runs when this file is
    # executed directly, e.g. `python model.py`.
    demo_model = build_model(15)
    demo_model.summary()
