"""
model.py
--------
CNN binary classifier (cat vs. dog) used in the SHAP attack experiments.

Architecture and training setup follow:
  Mollard, Becker, Röhrbein (2024).
  "Adversarial Evasion Attacks on Computer Vision using SHAP Values."
  DOI: 10.13140/RG.2.2.28762.40647
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Reshape, Flatten, Dropout, BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
from sklearn.model_selection import KFold


def build_classifier(input_shape: int = 12288, dropout: float = 0.5) -> Model:
    """
    Build a small CNN for binary image classification.

    Input is a flattened image vector (default: 64×64×3 = 12 288 values).
    Output is a single sigmoid unit (1 = cat, 0 = dog).

    Parameters
    ----------
    input_shape : int    — number of input features (flattened pixels)
    dropout     : float  — dropout rate applied before the output layer

    Returns
    -------
    Uncompiled Keras Model
    """
    inp = Input(shape=(input_shape,), name="image_input")
    x = Reshape((64, 64, 3))(inp)

    x = Conv2D(32,  kernel_size=(6, 6), strides=(2, 2), activation="relu", name="conv1")(x)
    x = Conv2D(64,  kernel_size=(3, 3), strides=(2, 2), activation="relu", name="conv2")(x)
    x = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation="relu", name="conv3")(x)

    x = Flatten()(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out, name="CatDogClassifier")


def train_classifier(images: np.ndarray, labels: np.ndarray,
                     folds: int = 3, epochs: int = 10,
                     batch_size: int = 48, lr: float = 1e-3) -> Model:
    """
    Train the classifier with k-fold cross-validation.

    Only the model from the last fold is returned (consistent with the
    original notebook). For production use, consider an ensemble or
    selecting the best fold by validation accuracy.

    Parameters
    ----------
    images     : np.ndarray, shape (N, 12288) — normalised images [0, 1]
    labels     : np.ndarray, shape (N,)       — binary labels (1=cat, 0=dog)
    folds      : int    — number of CV folds
    epochs     : int    — training epochs per fold
    batch_size : int    — mini-batch size
    lr         : float  — Adam learning rate

    Returns
    -------
    Trained Keras Model
    """
    loss = BinaryCrossentropy(label_smoothing=0.2)
    model = build_classifier()
    model.compile(optimizer=Adam(lr), loss=loss, metrics=["BinaryAccuracy"])
    model.summary()

    kf = KFold(n_splits=folds, shuffle=True, random_state=123)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(images), 1):
        print(f"\n── Fold {fold}/{folds} ──")
        model.fit(
            x=tf.convert_to_tensor(images[tr_idx]),
            y=labels[tr_idx],
            validation_data=(images[te_idx], labels[te_idx]),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )

    return model


if __name__ == "__main__":
    from data_utils import load_cat_dog

    images, labels = load_cat_dog()
    model = train_classifier(images, labels)
    model.save("cat_dog_classifier.keras")
    print("Model saved to cat_dog_classifier.keras")
