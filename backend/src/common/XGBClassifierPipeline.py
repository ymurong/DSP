import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from src.common.BasePipeline import BasePipeline
import datetime

logging.getLogger(__name__)


class XGBClassifierPipeline(BasePipeline):
    def __init__(self, model_file_name, model_training=False, model_params={},
                 **kwargs):
        super(XGBClassifierPipeline, self).__init__(model_file_name=model_file_name, model_training=model_training)
        if model_training:
            self.pipeline = None
        else:
            self.load_pipeline()
        self.model_params = model_params

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

    def plot_confusion_matrix(self, h=15, w=15):
        assert self.metrics is not None, "You must evaluate the model with test data before plotting the results"
        fig, ax = plt.subplots(figsize=(h, w))  # Sample figsize in inches
        sns.set(font_scale=4)
        sns.heatmap(self.metrics["confusion_matrix"], annot=True, linewidths=.5, fmt='g', ax=ax)
        plt.show()

    def predict_proba(self, X_test: pd.DataFrame):
        """
        :param X_test:
        :return: probability of being positive
        """
        return self.pipeline.predict_proba(X_test.copy())[:, 1]


if __name__ == '__main__':
    # load test data
    df_test = pd.read_csv("../resources/test_dataset_december.csv")
    columns = ['ip_node_degree', 'card_node_degree', 'email_node_degree', 'is_credit',
               'ip_address_woe', 'email_address_woe', 'card_number_woe', 'no_ip',
               'no_email', 'same_country', 'merchant_Merchant B',
               'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
               'card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa',
               'device_type_Linux', 'device_type_MacOS', 'device_type_Other',
               'device_type_Windows', 'device_type_iOS', 'shopper_interaction_POS']
    X_test = df_test[columns]
    y_test = df_test["has_fraudulent_dispute"]

    # evaluate
    pipeline = XGBClassifierPipeline(model_file_name="../resources/pretrained_models/xgboost_classifier_model.pkl")
    metrics = pipeline.eval(X_test, y_test)
    print(metrics)

    # plot confusion matrix
    pipeline.plot_confusion_matrix()

    # produce prediction probability results with psp_reference
    y_predict_proba = pd.Series(pipeline.predict_proba(X_test), name="predict_proba")
    df_pred_prob = pd.concat([df_test["psp_reference"], y_predict_proba], axis=1)
    df_pred_prob["created_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
    df_pred_prob["updated_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
    df_pred_prob.to_csv("../../predictions_dump.csv", index=False)
