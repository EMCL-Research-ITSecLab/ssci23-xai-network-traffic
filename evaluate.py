from tensorflow import keras

import config
# import plots

if __name__ == "__main__":
    current_config = config.config_multiclass_cnn
    model = keras.models.load_model("results/models/save_at_0_multiclass_cnn-flow-minp2-dim16-cols8-ALL-HEADER-ratio.keras")

    val_ds = keras.utils.image_dataset_from_directory(
        directory=current_config["dataset_path"] + "/test",
        labels="inferred",
        label_mode=current_config["label_mode"],
        color_mode="rgb",
        shuffle=True,
        batch_size=current_config["batch_size"],
        image_size=current_config["image_size"],
    )

    # plots.plot_images(val_ds, 128, 16)
    model.evaluate(val_ds)
