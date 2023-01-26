from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import dill as pickle
import os
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, recall_score, \
    precision_score, balanced_accuracy_score
from fairlearn.metrics import (
    false_positive_rate,
    false_negative_rate,
)
from fairlearn.metrics import MetricFrame

logging.getLogger(__name__)


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_file_name, model_training=False, **kwargs):
        self.model_file_name = model_file_name
        self.metrics = None
        self.fairness_metrics = None
        self.explainer = None

    @abstractmethod
    def explain(self, transaction_sample: np.ndarray, ) -> dict:
        pass

    @abstractmethod
    def get_influential_features(self, transaction_sample: np.ndarray, ) -> list:
        pass

    def load_explainer(self, explainer_file_name):
        dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(dir, explainer_file_name)
        with open(fname, 'rb') as handle:
            pickled_explainer = pickle.load(handle)
            self.explainer = pickled_explainer

    def predict_proba(self, X_test: pd.DataFrame):
        """
        :param X_test:
        :return: probability of being positive
        """
        return self.pipeline.predict_proba(X_test.copy())[:, 1]

    def predict(self, X_test: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        y_predict = (self.pipeline.predict_proba(X_test.copy())[:, 1] >= threshold).astype(bool)
        return y_predict

    def load_pipeline(self, **kwargs) -> None:
        dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(dir, self.model_file_name)
        with open(fname, 'rb') as handle:
            pickled_model = pickle.load(handle)
            self.pipeline = pickled_model

    def save_pipeline(self) -> None:
        dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(dir, self.model_file_name)
        with open(fname, 'wb') as handle:
            pickle.dump(self.pipeline, handle)

    def eval(self, X_test: pd.DataFrame, y_test: pd.DataFrame, threshold: float = 0.5):
        predicted = self.predict(X_test=X_test.copy(), threshold=threshold)
        self.metrics_sklearn(y_true=y_test, y_pred=predicted)
        return self.metrics

    def eval_fairness(self, X_test: pd.DataFrame, y_test: pd.DataFrame, A_test: pd.DataFrame, threshold: float = 0.5):
        predicted = self.predict(X_test=X_test.copy(), threshold=threshold)
        self.metrics_fairlearn(y_true=y_test, y_pred=predicted, sensitive_features=A_test)
        return self.fairness_metrics

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

    def metrics_fairlearn(self, y_true, y_pred, sensitive_features) -> Dict:
        fairness_metrics = {
            "balanced_accuracy": balanced_accuracy_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }
        metricframe_unmitigated = MetricFrame(
            metrics=fairness_metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        self.fairness_metrics = metricframe_unmitigated.by_group.to_dict()
        return self.fairness_metrics

    def plot_confusion_matrix(self, h=15, w=15):
        assert self.metrics is not None, "You must evaluate the model with test data before plotting the results"
        fig, ax = plt.subplots(figsize=(h, w))  # Sample figsize in inches
        sns.set(font_scale=4)
        sns.heatmap(self.metrics["confusion_matrix"], annot=True, linewidths=.5, fmt='g', ax=ax)
        plt.show()
