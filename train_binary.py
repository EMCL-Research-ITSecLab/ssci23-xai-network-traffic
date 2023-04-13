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
from metrics import WandbClfEvalCallback, PRMetrics
from models import custom_model, xception_model

train_ds, val_ds, test_ds = get_datasets(
    path=config.configs["dataset_path"],
    batch_size=config.configs["batch_size"],
    image_size=config.configs["image_size"],
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
    activation="sigmoid"
)


# Initialize a W&B run
run = wandb.init(
    project = config.configs["wandb_project"],
    name=config.configs["dataset_path"].split("/")[-1],
    config = config.configs
)

# keras.utils.plot_model(model, to_file="figures/model.pdf", show_shapes=True)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=3),
    keras.callbacks.ModelCheckpoint(filepath=config.configs["model_path"].format(config.configs["epochs"])),
    keras.callbacks.CSVLogger(config.configs["csv_path"], separator=",", append=False),
    keras.callbacks.TensorBoard('./logs', update_freq=1),
    # In case you want to use wandb
    # PRMetrics(val_ds, num_log_batches=config.configs["batch_size"]),
    WandbMetricsLogger(log_freq=10),
    WandbModelCheckpoint(filepath="models/"),
    WandbClfEvalCallback(
        val_ds,
        data_table_columns=["idx", "image", "ground_truth"],
        pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"]
    )
]

model.compile(
    optimizer=keras.optimizers.Adam(config.configs["learning_rate"]),
    loss=config.configs["loss"],
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
    epochs=config.configs["epochs"],
    callbacks=callbacks,
    validation_data=val_ds,
    validation_steps=500,
)

np.save(config.configs["history_path"].format(config.configs["epochs"]), history.history)

model = keras.models.load_model(config.configs["model_path"].format(config.configs["epochs"]))

# Close the W&B run
run.finish()