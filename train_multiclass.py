import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datasets import get_datasets
from metrics import f1_m, precision_m, recall_m
from models import xception_model

train_ds, val_ds, test_ds = get_datasets()

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

image_size = (128, 128)
batch_size = 128

model = xception_model(image_size + (3,), 17)

keras.utils.plot_model(model, show_shapes=True)

epochs = 2

callbacks = [
    # keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="loss", patience=3)
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", f1_m, precision_m, recall_m],
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
