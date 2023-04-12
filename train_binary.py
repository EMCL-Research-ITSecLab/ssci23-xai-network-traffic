import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# Weights and Biases related imports
import wandb
from tensorflow import keras
from wandb.keras import (WandbEvalCallback, WandbMetricsLogger,
                         WandbModelCheckpoint)

import config
from datasets import get_datasets
from models import custom_model, xception_model

train_ds, val_ds, test_ds = get_datasets(
    path=config.configs["dataset_path"],
    batch_size=config.configs["batch_size"],
    image_size=config.configs["image_size"] + (config.configs["image_channels"],),
    label_mode=config.configs["label_mode"],
    shuffle_buffer=config.configs["shuffle_buffer"],
    num_classes=config.configs["num_classes"]
)

# images, labels = tuple(zip(*dataset))

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

keras.backend.clear_session()

model = xception_model(
    config.configs["image_size"] + (config.configs["image_channels"],),
    config.configs["num_classes"],
)

keras.utils.plot_model(model, to_file="figures/model.pdf", show_shapes=True)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=3),
    keras.callbacks.ModelCheckpoint(filepath=config.configs["model_path"]),
    keras.callbacks.CSVLogger(config.configs["csv_path"], separator=",", append=False),
    WandbMetricsLogger(log_freq=10),
    WandbModelCheckpoint(filepath="models/"),
    WandbClfEvalCallback(
        validloader,
        data_table_columns=["idx", "image", "ground_truth"],
        pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"]
    )
]

model.compile(
    optimizer=keras.optimizers.Adam(config.configs["learning_rate"]),
    loss=config.configs["loss"],
    metrics=[
        keras.metrics.Accuracy(),
        keras.metrics.Precision(),
        keras.metrics.AUC(),
        keras.metrics.Recall(),
    ],
)

model.summary()

# Initialize a W&B run
run = wandb.init(
    project = config.configs["wandb_project"],
    config = config.configs
)

history = model.fit(
    train_ds,
    epochs=config.configs["epochs"],
    callbacks=callbacks,
    validation_data=val_ds,
    validation_steps=500,
)

np.save(config.configs["history_path"], history.history)

model = keras.models.load_model(config.configs["model_path"])

# Close the W&B run
run.finish()