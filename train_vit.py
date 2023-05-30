import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support, classification_report, accuracy_score)
from sklearn.utils import class_weight
from tensorflow import keras
import tensorflow as tf
import sys

import config
from datasets import get_datasets
from models import vit

if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    args = sys.argv[1:]
    if args[0] == "multiclass_vit":
        current_config = config.config_multiclass_vit
    elif args[0] == "binary_vit":
        current_config = config.config_binary_vit
    elif args[0] == "multiclass_vit_header":
        current_config = config.config_multiclass_vit_header
    elif args[0] == "binary_vit_header":
        current_config = config.config_binary_vit_header
    else:
        raise Exception(f"{args[0]} not defined")

    weight_decay = 0.0001

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

    model = vit(current_config)

    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=current_config["learning_rate"],
            weight_decay=weight_decay,
        ),
        loss=current_config["loss"],
        metrics=[
            # keras.metrics.Accuracy(),
            # keras.metrics.Precision(),
            # keras.metrics.Recall(),
            # keras.metrics.AUC(
            #     multi_label=True, num_labels=current_config["num_classes"]
            # ),
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=current_config["earlystopping_patience"]),
        keras.callbacks.ModelCheckpoint(
            filepath=current_config["model_path"].format(current_config["epochs"]),
            # save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            current_config["tensorboard_path"].format(current_config["epochs"]),
            update_freq=1,
        ),
    ]

    if current_config["type"] == "BINARY":
        # Class Weights for imbalanced data set
        ds_labels = [int(labels.numpy()) for _, labels in train_ds.unbatch()]
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=[i for i in range(0, current_config["num_classes"])],
            y=ds_labels,
        )
    else:
        ds_labels = [int(labels.numpy()) for _, labels in train_ds.unbatch()]
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

    history = model.fit(
        train_ds,
        batch_size=current_config["batch_size"],
        epochs=current_config["epochs"],
        callbacks=callbacks,
        validation_data=val_ds,
        validation_steps=1,
        class_weight=class_dict,
    )

    # model.load_weights(
    #     current_config["model_path"].format(
    #         current_config["epochs"]
    #     )
    # )

    model = keras.models.load_model(current_config["model_path"].format(current_config["epochs"]))

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
        y_pred = y_pred.argmax(axis=-1)

        dispaly_labels = display_labels = ["Benign", "Malware"]
    else:
        y_pred = y_pred.argmax(axis=-1)

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

    print(f"Y test has values {y_test[:5]}")
    print(f"Y pred has values {y_pred[:5]}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="all")
    cmd = ConfusionMatrixDisplay(cm, display_labels=dispaly_labels)
    cmd.plot()
    cmd.ax_.set(xlabel="Predicted", ylabel="True")
    plt.savefig(current_config["model_path"].format(current_config["epochs"]) + ".pdf")

    # Scores
    # precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)

    # print("precision: {}".format(precision))
    # print("recall: {}".format(recall))
    # print("fscore: {}".format(fscore))
    # print("support: {}".format(support))

    print(classification_report(y_test, y_pred, target_names=dispaly_labels))

    # print("Accuracy Score:")
    # print(accuracy_score(y_test, y_pred))
