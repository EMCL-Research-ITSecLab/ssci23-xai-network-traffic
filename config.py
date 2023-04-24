from tensorflow import keras

import wandb

wandb.login()

config_binary_cnn = dict(
    num_classes=2,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="binary",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=1,
    dataset_path="/home/smachmeier/data/binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed",
    history_path="results/save_at_{0}_binary_cnn_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.npy",
    model_path="results/save_at_{0}_binary_cnn_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.keras",
    csv_path="results/log_binary_cnn_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="binary-xception"
)

# TODO: Multiclass CNN training
config_multiclass_cnn = dict(
    num_classes=20,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="binary",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=3,
    dataset_path="/home/smachmeier/data/binary-flow-minp0-dim16-cols8-ALL-HEADER-split",
    history_path="results/save_at_{0}_vit.npy",
    model_path="results/save_at_{0}_vit.keras",
    csv_path="results/log.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="multiclass-xception"
)

config_binary_vit = dict(
    num_classes=2,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="binary",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=3,
    dataset_path="/home/smachmeier/data/binary-flow-minp2-dim16-cols8-filtered-by-hash-HEADER-split",
    history_path="results/save_at_{0}_vit.npy",
    model_path="results/save_at_{0}_vit.keras",
    csv_path="results/log.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="binary-vit"
)

# TODO: Multiclass ViT training
config_multiclass_vit = dict(
    num_classes=2,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="binary",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=3,
    dataset_path="/home/smachmeier/data/binary-flow-minp2-dim16-cols8-filtered-by-hash-HEADER-split",
    history_path="results/save_at_{0}_vit.npy",
    model_path="results/save_at_{0}_vit.keras",
    csv_path="results/log.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="multiclass-vit"
)
