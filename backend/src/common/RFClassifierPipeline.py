import pandas as pd
import logging
import numpy as np
from src.common.BasePipeline import BasePipeline
from src.resources.conf import INPUT_FEATURES, OUTPUT_FEATURE, EXPLAINABLE_CATEGORIES, SENSITIVE_FEATURE
import datetime

logging.getLogger(__name__)


class RFClassifierPipeline(BasePipeline):
    def __init__(self, model_file_name,
                 model_training=False, model_params={},
                 **kwargs):
        super(RFClassifierPipeline, self).__init__(model_file_name=model_file_name, model_training=model_training)
        self.model_params = model_params
        if model_training:
            self.pipeline = None
        else:
            self.load_pipeline()

    def explain(self, transaction_sample: np.ndarray, ) -> dict:
        def get_feature_name(feature_array_exp):
            for element in feature_array_exp:
                if element in INPUT_FEATURES:
                    return element
            return feature_array_exp[0]

        def get_explainable_group(feature_name):
            for key in EXPLAINABLE_CATEGORIES:
                if feature_name in EXPLAINABLE_CATEGORIES[key]:
                    return key
            return "general_evidences"

        if self.explainer is not None:
            predict_fn_rf = lambda x: self.pipeline.predict_proba(x).astype(float)
            exp = self.explainer.explain_instance(transaction_sample, predict_fn_rf, num_features=100)
            explanability_scores = {
                "ip_risk": 0.5,
                "email_risk": 0.5,
                "risk_card_behaviour": 0.5,
                "risk_card_amount": 0.5,
                "general_evidences": 0.5
            }
            for feature in exp.as_list():
                feature_name = get_feature_name(feature[0].split(" "))
                score = feature[1]
                explainable_group = get_explainable_group(feature_name)
                explanability_scores[explainable_group] += score
            return explanability_scores
        raise RuntimeError("explainer needs to be loaded first by invoking load_explainer method")
    
    def get_influential_features(self, transaction_sample: np.ndarray, ) -> list:
        def get_feature_name(feature_array_exp):
            for element in feature_array_exp:
                if element in INPUT_FEATURES:
                    return element
            return feature_array_exp[0]
        
        if self.explainer is not None:
            predict_fn_rf = lambda x: self.pipeline.predict_proba(x).astype(float)
            exp = self.explainer.explain_instance(transaction_sample, predict_fn_rf, num_features=100)
            influential_features = []
            for feature in exp.as_list():
                feature_name = get_feature_name(feature[0].split(" "))
                score = feature[1]
                if score >= 0:
                    influential_features.append(feature_name)
                    if len(influential_features) >= 5:
                        break
            return influential_features
        raise RuntimeError("explainer needs to be loaded first by invoking load_explainer method")



if __name__ == '__main__':
    # load test data
    df_test = pd.read_csv("../resources/test_dataset_december.csv")
    X_test = df_test[INPUT_FEATURES]
    y_test = df_test[OUTPUT_FEATURE]
    A_test = df_test[SENSITIVE_FEATURE]

    # evaluate
    pipeline = RFClassifierPipeline(model_file_name="../resources/pretrained_models/RandomForest.pickle")
    pipeline.load_explainer("../resources/pretrained_models/RandomForest_LIME.pickle")
    metrics = pipeline.eval(X_test, y_test)
    fairness_metrics = pipeline.eval_fairness(X_test, y_test, A_test)
    print({**metrics, **fairness_metrics})

    # plot confusion matrix
    pipeline.plot_confusion_matrix()

    # explain the model
    transaction_sample = X_test.sample().values[0]
    explanability_scores = pipeline.explain(transaction_sample)
    print(explanability_scores)

    # produce prediction probability results with psp_reference
    y_predict_proba = pd.Series(pipeline.predict_proba(X_test), name="predict_proba")
    df_pred_prob = pd.concat([df_test["psp_reference"], y_predict_proba], axis=1)
    df_pred_prob["created_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
    df_pred_prob["updated_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
    df_pred_prob.to_csv("../../predictions_dump_december.csv", index=False)
