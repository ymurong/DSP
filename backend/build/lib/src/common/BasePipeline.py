from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, \
    precision_score

logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_file_name, model_training=False, **kwargs):
        self.model_file_name = model_file_name
        self.metrics = None

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def load_pipeline(self):
        pass

    @abstractmethod
    def explain(self, transaction_sample: np.ndarray):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def metrics_sklearn(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        confusion_matrix_ = confusion_matrix(y_true, y_pred)
        false_neg = confusion_matrix_[1, 0]
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred),
            "block_rate": len(y_pred[y_pred == True]) / len(y_pred),
            "fraud_rate": false_neg / len(y_pred[y_pred == False]),
            "confusion_matrix": confusion_matrix_,
            "ks": max(abs(fpr - tpr)),
        }
        return self.metrics


