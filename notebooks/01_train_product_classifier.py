"""
Transfer Learning Script: MobileNetV2 Product Image Classifier.

Categories (5): ['bags', 'clothing', 'electronics', 'groceries', 'shoes']
Artifact Output: models/product_classifier.h5
"""

import os
from pathlib import Path
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CATEGORIES = ["bags", "clothing", "electronics", "groceries", "shoes"]
IMAGE_SIZE = (224, 224)
MODEL_SAVE_PATH = Path(__file__).resolve().parent.parent / "models" / "product_classifier.h5"


def build_transfer_learning_model(num_classes: int = 5):
    """
    Build a Transfer Learning model using MobileNetV2 backbone.

    :param num_classes: Number of product output categories.
    :return: Compiled tf.keras Model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2
    except ImportError:
        raise ImportError(
            "TensorFlow is required to build the Product Image Classifier."
        )

    # Base model with pre-trained ImageNet weights (excluding top classification head)
    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model layers for feature extraction
    base_model.trainable = False

    # Classification head
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    # MobileNetV2 preprocessing: scales pixel values to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Product_Classifier")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_and_save_model():
    """
    Builds, initializes weights, and exports the product classifier model to HDF5.
    """
    print(f"Building MobileNetV2 Transfer Learning Model for {len(CATEGORIES)} categories: {CATEGORIES}...")
    model = build_transfer_learning_model(num_classes=len(CATEGORIES))
    model.summary()

    # Ensure output directory exists
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving compiled model to: {MODEL_SAVE_PATH}")
    model.save(str(MODEL_SAVE_PATH))
    print("Model successfully exported to models/product_classifier.h5!")


if __name__ == "__main__":
    train_and_save_model()
