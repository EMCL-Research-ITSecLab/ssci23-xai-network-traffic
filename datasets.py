from tensorflow import keras
import tensorflow as tf


def get_datasets(path, batch_size, image_size, label_mode, shuffle_buffer, num_classes, labels="inferred", color_mode="rgb"):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        directory=f"{path}/train",
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        directory=f"{path}/val",
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        directory=f"{path}/test",
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
    )

    return train_ds, val_ds, test_ds
