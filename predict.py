import glob

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from tensorflow import keras

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


# model = keras.models.load_model('save_at_1.keras')

# for filename in glob.iglob("/home/smachmeier/data/binary-classification-flow-minp3-dim16-cols8-split/test/benign/"+ '*.png', recursive=True):
#    img = tf.keras.utils.load_img(filename)
#    input_arr = tf.keras.utils.img_to_array(img)
#    input_arr = np.array([input_arr])  # Convert single image to a batch.
#    print(model.predict(input_arr))
#    break

# for filename in glob.iglob("/home/smachmeier/data/test-images/"+ '*.png', recursive=True):
#    img = tf.keras.utils.load_img(filename)
#    input_arr = tf.keras.utils.img_to_array(img)
#    input_arr = np.array([input_arr])  # Convert single image to a batch.
#    print(model.predict(input_arr))
