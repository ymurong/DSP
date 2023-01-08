from abc import ABCMeta, abstractmethod
import pandas as pd
from xgboost import XGBClassifier
import logging
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, \
    precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_file_name, model_training=False, **kwargs):
        self.model_file_name = model_file_name
        self.metrics = {}

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def load_pipeline(self):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def metrics_sklearn(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "ks": max(abs(fpr - tpr)),
        }
        return self.metrics



