from tensorflow import keras

import config
import plots

if __name__ == "__main__":
    model = keras.models.load_model("results/save_at_10_binary_good_results.keras")

    val_ds = keras.utils.image_dataset_from_directory(
        directory=config.config_binary_cnn["dataset_path"] + "/test",
        labels="inferred",
        label_mode=config.config_binary_cnn["label_mode"],
        color_mode="rgb",
        shuffle=True,
        batch_size=config.config_binary_cnn["batch_size"],
        image_size=config.config_binary_cnn["image_size"],
    )

    plots.plot_images(val_ds, 128, 16)
    model.evaluate(val_ds)
