import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from scikitplot.metrics import plot_confusion_matrix, plot_roc


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self,val_x, val_y, batch_size = 20):
        super().__init__()
        self.val_x = val_x
        self.val_y = val_y
        self.batch_size = batch_size
    def on_train_begin(self, logs=None):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
    def on_epoch_end(self, epoch, logs=None):
        x    = self.val_x
        targ = self.val_y
        score   = np.asarray(self.model.predict(x))
        predict = np.round(np.asarray(self.model.predict(x)))
        self.f1s.append(f1_score(targ, predict, average='micro'))
        self.confusion.append(confusion_matrix(targ.argmax(axis=1), predict.argmax(axis=1)))
        print("\nAt epoch {} f1_score {}:".format(epoch, self.f1s[-1]))
        print('\nAt epoch {} cm {}'.format(epoch, self.confusion[-1]))
        return 

class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)

class PerformanceVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16,12))
        plot_roc(y_true, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))