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


class WeightedCategoricalCrossentropy(keras.losses.CategoricalCrossentropy):

    def __init__(
        self,
        weights,
        from_logits=False,
        label_smoothing=0,
        reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name='categorical_crossentropy',
    ):
        super().__init__(
            from_logits, label_smoothing, reduction, name=f"weighted_{name}"
        )
        self.weights = weights

    def call(self, y_true, y_pred):
        weights = self.weights
        nb_cl = len(weights)
        final_mask = keras.backend.zeros_like(y_pred[:, 0])
        y_pred_max = keras.backend.max(y_pred, axis=1)
        y_pred_max = keras.backend.reshape(
            y_pred_max, (keras.backend.shape(y_pred)[0], 1))
        y_pred_max_mat = keras.backend.cast(
            keras.backend.equal(y_pred, y_pred_max), keras.backend.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (
                weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return super().call(y_true, y_pred) * final_mask

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
    # keras.callbacks.CSVLogger(config.configs["csv_path"], separator=",", append=False),
    # keras.callbacks.TensorBoard('./logs', update_freq=1),
    # In case you want to use wandb
    # PRMetrics(val_ds, num_log_batches=config.configs["batch_size"]),
    # WandbMetricsLogger(log_freq=10),
    # WandbModelCheckpoint(filepath="models/"),
    # WandbClfEvalCallback(
    #     val_ds,
    #     data_table_columns=["idx", "image", "ground_truth"],
    #     pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"]
    # )
]

ds_labels = [int(labels.numpy()[0]) for _, labels in train_ds.unbatch()]
class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=[0,1], y=ds_labels)
print(class_weights)
class_dict = {0: class_weights[0], 1:class_weights[1]}
model.compile(
    optimizer=keras.optimizers.Adam(config.configs["learning_rate"]),
    # loss=WeightedCategoricalCrossentropy(class_weights),
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
    class_weight=class_dict
)

np.save(config.configs["history_path"].format(config.configs["epochs"]), history.history)

model = keras.models.load_model(config.configs["model_path"].format(config.configs["epochs"]))

# Close the W&B run
run.finish()