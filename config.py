import wandb
from tensorflow import keras

wandb.login()

configs = dict(
    num_classes=1,
    shuffle_buffer=1024,
    batch_size=256,
    image_size=(128, 128),
    image_channels=3,
    label_mode="binary",
    earlystopping_patience=3,
    learning_rate=1e-3,
    epochs=2,
    dataset_path="/home/smachmeier/data/binary-flow-minp3-dim16-cols8-split",
    history_path="results/save_at_{0}.npy",
    model_path="results/save_at_{0}.keras",
    csv_path="log.csv",
    loss=keras.losses.BinaryCrossentropy(),
    wandb_project="binary-xception"
)
