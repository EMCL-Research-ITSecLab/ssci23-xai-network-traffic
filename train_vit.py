from tensorflow import keras
import wandb

import config
from datasets import get_datasets
from models import vit

if __name__ == "__main__":
    current_config = config.config_binary_vit

    weight_decay = 0.0001

    if current_config["wandb_active"]:
        wandb.login()

    train_ds, val_ds, test_ds = get_datasets(
        path=config.config_binary_vit["dataset_path"],
        batch_size=config.config_binary_vit["batch_size"],
        image_size=config.config_binary_vit["image_size"],
        label_mode="int",
        shuffle_buffer=config.config_binary_vit["shuffle_buffer"],
        num_classes=config.config_binary_vit["num_classes"],
    )

    model = vit(current_config)

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=config.config_binary_vit["learning_rate"],
            weight_decay=weight_decay,
        ),
        # loss=config.config_binary_vit["loss"],
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            # "accuracy",
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.Precision(),
            # keras.metrics.AUC(),
            # keras.metrics.Recall(),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.config_binary_vit["model_path"].format(
                config.config_binary_vit["epochs"]
            ),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_ds,
        batch_size=config.config_binary_vit["batch_size"],
        epochs=config.config_binary_vit["epochs"],
        callbacks=callbacks,
        validation_data=val_ds,
    )

    model.load_weights(
        config.config_binary_vit["model_path"].format(
            config.config_binary_vit["epochs"]
        )
    )

    _, accuracy, top_5_accuracy = model.evaluate(test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
