import wandb
from tensorflow import keras

wandb.login()

configs = dict(
    num_classes=10,
    shuffle_buffer=1024,
    batch_size=64,
    image_size=(128, 128),
    image_channels=3,
    label_mode="categorical",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=10,
    dataset_path="",
    history_path=f"results/save_at_{epoch}.npy",
    model_path=f"results/save_at_{epoch}.keras",
    csv_path="log.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project=""
)
