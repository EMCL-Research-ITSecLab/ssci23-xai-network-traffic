# Weights and Biases related imports
import numpy as np
import tensorflow as tf
from wandb.keras import WandbEvalCallback

import wandb


class PRMetrics(tf.keras.callbacks.Callback):
  """ Custom callback to compute per-class PR & ROC curves
  at the end of each training epoch"""
  def __init__(self, generator=None, num_log_batches=1):
    self.generator = generator
    self.num_batches = num_log_batches
    # store full names of classes
    self.class_names = { v: k for k, v in generator.class_indices.items() }
    self.flat_class_names = [k for k, v in generator.class_indices.items()]

  def on_epoch_end(self, epoch, logs={}):
    # collect validation data and ground truth labels from generator
    val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
    val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

    # use the trained model to generate predictions for the given number
    # of validation data batches (num_batches)
    val_predictions = self.model.predict(val_data)
    ground_truth_class_ids = val_labels.argmax(axis=1)

    # Log precision-recall curve
    # the key "pr_curve" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"roc_curve" : wandb.plot.roc_curve(ground_truth_class_ids, val_predictions, labels=self.flat_class_names)})


class ConfusionMetrics(tf.keras.callbacks.Callback):
  """ Custom callback to compute metrics at the end of each training epoch"""
  def __init__(self, generator=None, num_log_batches=1):
    self.generator = generator
    self.num_batches = num_log_batches
    # store full names of classes
    self.flat_class_names = [k for k, v in generator.class_indices.items()]

  def on_epoch_end(self, epoch, logs={}):
    # collect validation data and ground truth labels from generator
    val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
    val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

    # use the trained model to generate predictions for the given number
    # of validation data batches (num_batches)
    val_predictions = self.model.predict(val_data)
    ground_truth_class_ids = val_labels.argmax(axis=1)
    # take the argmax for each set of prediction scores
    # to return the class id of the highest confidence prediction
    top_pred_ids = val_predictions.argmax(axis=1)

    # Log confusion matrix
    # the key "conf_mat" is the id of the plot--do not change
    # this if you want subsequent runs to show up on the same plot
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            preds=top_pred_ids, y_true=ground_truth_class_ids,
                            class_names=self.flat_class_names)})


class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validloader, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.val_data = validloader.unbatch().take(num_samples)

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            self.data_table.add_data(idx, wandb.Image(image), np.argmax(label, axis=-1))

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()

        for idx in table_idxs:
            pred = preds[idx]
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                pred,
            )

    def _inference(self):
        preds = []
        for image, label in self.val_data:
            pred = self.model(tf.expand_dims(image, axis=0))
            argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
            preds.append(argmax_pred)

        return preds
