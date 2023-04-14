import glob

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.utils import class_weight
from tensorflow import keras

import config

model = keras.models.load_model('results/save_at_10.keras')

val_ds = keras.utils.image_dataset_from_directory(
    directory="/home/smachmeier/data/test-data-flow-minp3-dim16-cols8",
    labels="inferred",
    label_mode=config.configs["label_mode"],
    color_mode="rgb",
    shuffle=False,
    batch_size=config.configs["batch_size"],
    image_size=config.configs["image_size"],
)

# ds_labels = [int(labels.numpy()[0]) for _, labels in val_ds.unbatch()]
# print(ds_labels)
# class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=[0,1], y=ds_labels)
# print(class_weights)
# model.fit(val_ds,epochs=10)

print(model.evaluate(val_ds))

# def predict_binary():
#     # TODO Predict test set
#     keras_model = keras.models.load_model('path/to/location')

#     y_pred_keras = keras_model.predict(X_test).ravel()
#     fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

#     auc_keras = auc(fpr_keras, tpr_keras)

# def predict_multiclass():
#     # TODO Predict test set
#     keras_model = keras.models.load_model('path/to/location')

#     y_pred_keras = keras_model.predict(X_test).ravel()
#     fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

#     auc_keras = auc(fpr_keras, tpr_keras)

# def draw_confusion_matrix(true, preds):
#    conf_matx = confusion_matrix(true, preds)
#    sns.heatmap(
#       conf_matx, 
#       annot=True, 
#       annot_kws={"size": 12},
#       fmt='g', 
#       cbar=False, 
#       cmap="viridis"
#    )
#    plt.savefig("Test.png")

# def convert_to_labels(preds_array):
#     preds_df = pd.DataFrame(preds_array)
#     predicted_labels = preds_df.idxmax(axis=1)
#     return predicted_labels

# def preprocess_image(img):
#    img = img.resize((224, 224))
#    img = np.array(img)
#    img = np.expand_dims(img, axis=0)
#    img = preprocessing_function(img)
#    return img

# def make_prediction(model, img_path):
#    image = Image.open(img_path)
#    img_preprocessed = preprocess_image(image)
#    prediction = np.argmax(model.predict(img_preprocessed), axis=-1)[0]
#    return prediction
