from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow import keras
from wandb.keras import (WandbEvalCallback, WandbMetricsLogger,
                         WandbModelCheckpoint)

import config
# Weights and Biases related imports
import wandb
from datasets import get_datasets
from metrics import PRMetrics, WandbClfEvalCallback
from models import custom_model, xception_model

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets(
        path=config.config_binary_cnn["dataset_path"],
        batch_size=config.config_binary_cnn["batch_size"],
        image_size=config.config_binary_cnn["image_size"],
        label_mode=config.config_binary_cnn["label_mode"],
        shuffle_buffer=config.config_binary_cnn["shuffle_buffer"],
        num_classes=config.config_binary_cnn["num_classes"]
    )

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = xception_model(
        config.config_binary_cnn["image_size"] + (config.config_binary_cnn["image_channels"],),
        config.config_binary_cnn["num_classes"],
        activation="sigmoid"
    )

    # Initialize a W&B run
    run = wandb.init(
        project = config.config_binary_cnn["wandb_project"],
        name=config.config_binary_cnn["dataset_path"].split("/")[-1],
        config = config.config_binary_cnn
    )

    keras.utils.plot_model(model, to_file="figures/model.pdf", show_shapes=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=3),
        keras.callbacks.ModelCheckpoint(filepath=config.config_binary_cnn["model_path"].format(config.config_binary_cnn["epochs"])),
        # keras.callbacks.CSVLogger(config.config_binary_cnn["csv_path"], separator=",", append=False),
        # keras.callbacks.TensorBoard('./logs', update_freq=1),
        # In case you want to use wandb
        # PRMetrics(val_ds, num_log_batches=config.config_binary_cnn["batch_size"]),
        # WandbMetricsLogger(log_freq=10),
        # WandbModelCheckpoint(filepath="models/"),
        # WandbClfEvalCallback(
        #     val_ds,
        #     data_table_columns=["idx", "image", "ground_truth"],
        #     pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"]
        # )
    ]

    ds_labels = [int(labels.numpy()[0]) for _, labels in train_ds.unbatch()]
    print(f"Labes: {ds_labels[:25]}")

    class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=[0,1], y=ds_labels)
    print(f"Class weights are {class_weights}")

    class_dict = {0: class_weights[0], 1:class_weights[1]}
    print(f"Class dict are {class_dict}")

    model.compile(
        optimizer=keras.optimizers.Adam(config.config_binary_cnn["learning_rate"]),
        # loss=WeightedCategoricalCrossentropy(class_weights),
        loss=config.config_binary_cnn["loss"],
        metrics=[
            "accuracy",
            keras.metrics.Precision(),
            keras.metrics.AUC(),
            keras.metrics.Recall(),
        ],
    )

    model.summary()

    history = model.fit(
        train_ds,
        epochs=config.config_binary_cnn["epochs"],
        callbacks=callbacks,
        validation_data=val_ds,
        validation_steps=500,
        class_weight=class_dict
    )

    np.save(config.config_binary_cnn["history_path"].format(config.config_binary_cnn["epochs"]), history.history)

    model = keras.models.load_model(config.config_binary_cnn["model_path"].format(config.config_binary_cnn["epochs"]))

    print("Model evaluation:")
    model.evaluate(test_ds)

    # Close the W&B run
    run.finish()
