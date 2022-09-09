import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import tensorflow as tf

#tf.config.run_functions_eagerly(True)

AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10
BATCH_SIZE = 32
DATASET = "/home/tz251/Desktop/3_Png"
IMG_HEIGHT = 180
IMG_WIDTH = 180
# DATASET = "/home/tz251/Documents/DPI/USTC-TK2016/4_Png"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)


def macro_f1(y, y_hat, thresh=0.5):
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
DATASET = pathlib.Path(data_dir)

image_count = len(list(DATASET.glob('*/*.jpg')))
print(image_count)


# TODO: Data is resized to (28,28). This should be changed.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # DATASET + "/Train",
    DATASET,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # DATASET + "/Test",
    DATASET,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
)

class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.math.argmax(labels[i])])
        print(labels[i])
        plt.axis("off")
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, 
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Constant(0.1), 
        # strides=(1, 1), 
        padding='SAME', 
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, 
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Constant(0.1), 
        padding='SAME', 
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Constant(0.1), 
        # strides=(1, 1), 
        padding='SAME', 
        activation='relu'
    ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Constant(0.1), 
        activation='relu'
    ),
    tf.keras.layers.Dense(num_classes, name="outputs")
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', macro_f1]
)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Save the model
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=5*BATCH_SIZE,
                                                 verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
    # callbacks=[cp_callback]
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
plt.show()
