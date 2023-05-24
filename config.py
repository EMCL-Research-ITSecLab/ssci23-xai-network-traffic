from tensorflow import keras

DEFAULT_MODEL_PATH = "results/models"
DEFAULT_DATA_PATH = "/home/smachmeier/data"

config_binary_cnn = dict(
    num_classes=1,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="binary",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=25,
    dataset_path=f"{DEFAULT_DATA_PATH}/binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed",
    history_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_binary_cnn_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.npy",
    model_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_binary_cnn_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.keras",
    csv_path=f"{DEFAULT_MODEL_PATH}/log_binary_cnn_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="binary-xception",
    activation="sigmoid",
    print_model=False,
    wandb_active=False
)

config_multiclass_cnn = dict(
    num_classes=19,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="categorical",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=10,
    dataset_path=f"{DEFAULT_DATA_PATH}/multiclass-flow-minp2-dim16-cols8-ALL-HEADER-split-ratio",
    history_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_multiclass_cnn_flow-minp2-dim16-cols8-ALL-HEADER-ratio.npy",
    model_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_multiclass_cnn-flow-minp2-dim16-cols8-ALL-HEADER-ratio.keras",
    csv_path=f"{DEFAULT_MODEL_PATH}/log_multiclass_cnn-flow-minp2-dim16-cols8-ALL-HEADER-ratio.csv",
    loss=keras.losses.CategoricalCrossentropy(),
    wandb_project="multiclass-xception",
    activation="softmax",
    print_model=False,
    wandb_active=False
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
    epochs=25,
    dataset_path=f"{DEFAULT_DATA_PATH}/binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed",
    history_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_binary_vit_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.npy",
    model_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_binary_vit_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.keras",
    csv_path=f"{DEFAULT_MODEL_PATH}/log_binary_vit_binary-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="binary-vit",
    activation="sigmoid",
    print_model=False,
    wandb_active=False
)

config_multiclass_vit = dict(
    num_classes=20,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="int",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=25,
    dataset_path=f"{DEFAULT_DATA_PATH}/multiclass-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed",
    history_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_multiclass_vit_flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.npy",
    model_path=f"{DEFAULT_MODEL_PATH}/save_at_{0}_multiclass_vit-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.keras",
    csv_path=f"{DEFAULT_MODEL_PATH}/log_multiclass_vit-flow-minp2-dim16-cols8-ALL-HEADER-split-fixed.csv",
    loss=keras.losses.SparseCategoricalCrossentropy(),
    wandb_project="multiclass-vit",
    activation="softmax",
    print_model=False,
    wandb_active=False
)
