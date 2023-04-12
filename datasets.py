from tensorflow import keras

PATH="/home/smachmeier/data/binary-classification-flow-minp3-dim16-cols8-split"

def get_datasets(path=PATH, labels="inferred", label_mode="binary", color_mode="rgb", batch_size=32, image_size=(128,128)):
    train_ds = keras.utils.image_dataset_from_directory(
        directory=f"{PATH}/train",
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        directory=f"{PATH}/val",
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        directory=f"{PATH}/test",
        labels=labels,
        label_mode=label_mode,
        color_mode=color_mode,
        seed=1337,
        batch_size=batch_size,
        image_size=image_size,
    )

    return train_ds, val_ds, test_ds
