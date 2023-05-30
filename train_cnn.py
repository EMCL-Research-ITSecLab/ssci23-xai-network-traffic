import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils import class_weight
from tensorflow import keras
from wandb.keras import (WandbEvalCallback, WandbMetricsLogger,
                         WandbModelCheckpoint)

import config
from datasets import get_datasets
from metrics import PRMetrics, WandbClfEvalCallback

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "multiclass_cnn_vgg19":
        current_config = config.config_multiclass_cnn_vgg19
    elif args[0] == "multiclass_cnn_xception":
        current_config = config.config_multiclass_cnn_xception
    elif args[0] == "binary_cnn_xception":
        current_config = config.config_binary_cnn_xception
    elif args[0] == "binary_cnn_vgg19":
        current_config = config.config_binary_cnn_vgg19
    else:
        raise Exception(f"{args[0]} not defined")

    if current_config["wandb_active"]:
        wandb.login()

    train_ds, val_ds, test_ds = get_datasets(
        path=current_config["dataset_path"],
        batch_size=current_config["batch_size"],
        image_size=current_config["image_size"],
        label_mode=current_config["label_mode"],
        shuffle_buffer=current_config["shuffle_buffer"],
        num_classes=current_config["num_classes"],
    )

    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Create model
    model = current_config["model"](
        image_size=current_config["image_size"] + (current_config["image_channels"],),
        classes=current_config["num_classes"],
        activation=current_config["activation"],
    )

    if current_config["wandb_active"]:
        # Initialize a W&B run
        run = wandb.init(
            project=current_config["wandb_project"],
            name=current_config["dataset_path"].split("/")[-1],
            config=current_config,
        )

    if current_config["print_model"]:
        keras.utils.plot_model(
            model,
            to_file=current_config["model_path"].format(current_config["epochs"])
            + ".pdf",
            show_shapes=True,
        )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=current_config["earlystopping_patience"]),
        keras.callbacks.ModelCheckpoint(
            filepath=current_config["model_path"].format(current_config["epochs"]),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(
            current_config["csv_path"].format(current_config["epochs"]),
            separator=",",
            append=False,
        ),
        keras.callbacks.TensorBoard(
            current_config["tensorboard_path"].format(current_config["epochs"]),
            update_freq=1,
        ),
    ]

    # TODO: Integrate WanDB for Model publication
    if current_config["wandb_active"]:
        wandb_callback = [
            PRMetrics(val_ds, num_log_batches=current_config["batch_size"]),
            WandbMetricsLogger(log_freq=10),
            WandbModelCheckpoint(filepath="models/wandb"),
            WandbClfEvalCallback(
                val_ds,
                data_table_columns=["idx", "image", "ground_truth"],
                pred_table_columns=[
                    "epoch",
                    "idx",
                    "image",
                    "ground_truth",
                    "prediction",
                ],
            ),
        ]
        callbacks.append(wandb_callback)

    if current_config["type"] == "BINARY":
        # Class Weights for imbalanced data set
        ds_labels = [int(labels.numpy()[0]) for _, labels in train_ds.unbatch()]
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=[i for i in range(0, current_config["num_classes"] + 1)],
            y=ds_labels,
        )
    else:
        ds_labels = [int(np.argmax(labels.numpy())) for _, labels in train_ds.unbatch()]
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=[
                i for i in range(0, current_config["num_classes"])
            ],
            y=ds_labels,
        )

    print(f"Class weights are {class_weights}")
    class_dict = {}
    for index, classw in enumerate(class_weights):
        class_dict[index] = classw
    print(f"Class dict are {class_dict}")

    model.compile(
        optimizer=keras.optimizers.SGD(current_config["learning_rate"]),
        loss=current_config["loss"],
        metrics=[
            "accuracy",
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC(
                multi_label=True, num_labels=current_config["num_classes"]
            ),
        ],
    )
    if current_config["print_model"]:
        model.summary()

    history = model.fit(
        train_ds,
        epochs=current_config["epochs"],
        callbacks=callbacks,
        validation_data=val_ds,
        validation_steps=1,
        class_weight=class_dict,
    )

    if current_config["wandb_active"]:
        # Close the W&B run
        run.finish()

    np.save(
        current_config["history_path"].format(current_config["epochs"]),
        history.history,
    )

    #################################################################################
    ## EVALUATE
    #################################################################################

    model.load_weights(current_config["model_path"].format(current_config["epochs"]))

    print("Model evaluation:")
    model.evaluate(test_ds)

    # Important: Unbatching is necessary to get the correct order of images and labels
    x_test = []
    y_test = []
    for images, labels in test_ds.unbatch():
        y_test.append(labels.numpy())
        x_test.append(images)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Get Y prediction
    y_pred = model.predict(x_test)

    print(f"Y test has shape {y_test.shape}")
    print(f"Y pred has shape {y_pred.shape}")

    if current_config["type"] == "BINARY":
        y_pred = np.round(y_pred)
        y_pred = y_pred.reshape(-1).astype(int)

        dispaly_labels = display_labels = ["Benign", "Malware"]
    else:
        y_pred = y_pred.argmax(axis=-1)
        y_test = y_test.argmax(axis=-1)

        dispaly_labels = [
            "BitTorrent",
            "Cridex",
            "FTP",
            "Geodo",
            "Gmail",
            "Htbot",
            "Miuref",
            "MySQL",
            "Neris",
            "Nsis-ay",
            "Outlook",
            "Shifu",
            "Skype",
            "SMB",
            "Tinba",
            "Virut",
            "Weibo",
            "WorldOfWarcraft",
            "Zeus",
        ]

    print(f"Y test has values {y_test[5]}")
    print(f"Y pred has values {y_pred[5]}")
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="all")
    cmd = ConfusionMatrixDisplay(cm, display_labels=dispaly_labels)
    cmd.plot()
    cmd.ax_.set(xlabel="Predicted", ylabel="True")
    plt.savefig(current_config["model_path"].format(current_config["epochs"]) + ".pdf")

    # Scores
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("fscore: {}".format(fscore))
    print("support: {}".format(support))
