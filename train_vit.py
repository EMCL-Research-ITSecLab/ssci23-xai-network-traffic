import glob
import hashlib
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import config
from datasets import get_datasets

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

input_shape = (128, 128, 3)

train_ds, val_ds, test_ds = get_datasets(
    path=config.configs["dataset_path"],
    batch_size=config.configs["batch_size"],
    image_size=config.configs["image_size"],
    label_mode="int",
    shuffle_buffer=config.configs["shuffle_buffer"],
    num_classes=config.configs["num_classes"]
)

weight_decay = 0.0001
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (config.configs["image_size"][0] // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(config.configs["num_classes"])(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    model.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=config.configs["learning_rate"], weight_decay=weight_decay
        ),
        # loss=config.configs["loss"],
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            # "accuracy",
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.Precision(),
            # keras.metrics.AUC(),
            # keras.metrics.Recall(),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.configs["model_path"].format(config.configs["epochs"]),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_ds,
        batch_size=config.configs["batch_size"],
        epochs=config.configs["epochs"],
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.load_weights(config.configs["model_path"].format(config.configs["epochs"]))
    _, accuracy, top_5_accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
# history = run_experiment(vit_classifier)
