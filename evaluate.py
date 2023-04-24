import glob

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.utils import class_weight
from tensorflow import keras

import config
import plots

model = keras.models.load_model('results/save_at_10_binary_good_results.keras')

val_ds = keras.utils.image_dataset_from_directory(
    directory="/home/smachmeier/data/binary-flow-minp0-dim16-cols8-ALL-NONE-split-fixed/test",
    labels="inferred",
    label_mode=config.config_binary_cnn["label_mode"],
    color_mode="rgb",
    shuffle=True,
    batch_size=config.config_binary_cnn["batch_size"],
    image_size=config.config_binary_cnn["image_size"],
)

plots.plot_images(val_ds, 128, 16)
model.evaluate(val_ds)
