import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 3
BATCH_SIZE = 2048

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

#tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
#tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

dict_2class = {0: 'Benign', 1: 'Malware'}
dict_10class_benign = {0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail',
                       4: 'MySQL', 5: 'Outlook', 6: 'Skype', 7: 'SMB', 8: 'Weibo', 9: 'WorldOfWarcraft'}
dict_10class_malware = {0: 'Cridex', 1: 'Geodo', 2: 'Htbot', 3: 'Miuref',
                        4: 'Neris', 5: 'Nsis-ay', 6: 'Shifu', 7: 'Tinba', 8: 'Virut', 9: 'Zeus'}
dict_20class = {0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail', 4: 'MySQL', 5: 'Outlook', 6: 'Skype', 7: 'SMB', 8: 'Weibo', 9: 'WorldOfWarcraft',
                10: 'Cridex', 11: 'Geodo', 12: 'Htbot', 13: 'Miuref', 14: 'Neris', 15: 'Nsis-ay', 16: 'Shifu', 17: 'Tinba', 18: 'Virut', 19: 'Zeus'}
dict = {}


def get_dataset(get_info: bool = False):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/home/tz251/Desktop/3_Png/AllLayers',
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        seed=42,
        image_size=(256, 256),
        shuffle=True,
        validation_split=0.2,
        subset="training",
        interpolation="bilinear",
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/home/tz251/Desktop/3_Png/AllLayers',
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        seed=42,
        image_size=(256, 256),
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        interpolation="bilinear",
    )

    if get_info:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")
        # plt.show()

    return train_ds, validation_ds


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, (8,8), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (8,8), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (8,8), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(num_classes, name="outputs")
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model


train_ds, validation_ds = get_dataset()
class_names = train_ds.class_names

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

print(class_names)
num_classes = len(class_names)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=5*BATCH_SIZE,
                                                 verbose=1)

model = get_model()
#model.summary()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[cp_callback]
)

model.save('my_model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# plt.show()
