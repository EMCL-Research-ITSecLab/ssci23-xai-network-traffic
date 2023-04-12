import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from tensorflow import keras
from tensorflow.keras import layers

from datasets import get_datasets
from metrics import (PerformanceVisualizationCallback, f1_m, precision_m,
                     recall_m)
from models import custom_model, xception_model

train_ds, val_ds, test_ds = get_datasets()

# y_test = np.concatenate([y for x, y in test_ds], axis=0)
# x_test = np.concatenate([x for x, y in test_ds], axis=0)
# images, labels = tuple(zip(*dataset))

# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

image_size = (128, 128)
batch_size = 128

model = xception_model(image_size + (3,), 1)

keras.utils.plot_model(model, to_file="figures/model.pdf", show_shapes=True)

epochs = 1

callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=3),
    keras.callbacks.ModelCheckpoint(filepath="save_at_{epoch}.keras"),
    keras.callbacks.CSVLogger('./log.csv', separator=",", append=False),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall()],
)

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# np.save('binary_classification.npy', history.history)
model = keras.models.load_model('save_at_1.keras')

# print(model.evaluate(test_ds, batch_size=256))

# plt.figure(1)
# for x,y in test_ds:
#     y_pred_keras = model.predict(x)
#     print(np.argmax(y, axis=1))
#     print(np.argmax(y_pred_keras, axis=1))
#     break
#     B = np.where(y_pred_keras > 0.5, 1, 0)
#     # fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_test, axis=1), np.argmax(y_pred_keras, axis=1))

#     fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, B)

#     n_classes = 2
#     # print(y_pred_keras)
#     # print(test_label)
#     # fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_label, y_pred_keras)
#     auc_keras = auc(fpr_keras, tpr_keras)


#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     plt.savefig("test.png")
