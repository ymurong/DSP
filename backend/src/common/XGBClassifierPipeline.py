import pandas as pd
import logging
from src.common.BasePipeline import BasePipeline
from src.resources.conf import INPUT_FEATURES, OUTPUT_FEATURE
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

    def explain(self):
        raise NotImplementedError


if __name__ == '__main__':
    # load test data
    df_test = pd.read_csv("../resources/test_dataset_december.csv")
    X_test = df_test[INPUT_FEATURES]
    y_test = df_test[OUTPUT_FEATURE]

    # evaluate
    pipeline = XGBClassifierPipeline(model_file_name="../resources/pretrained_models/XGBoost.pickle")
    metrics = pipeline.eval(X_test, y_test)
    print(metrics)

    # plot confusion matrix
    pipeline.plot_confusion_matrix()

    # produce prediction probability results with psp_reference
    y_predict_proba = pd.Series(pipeline.predict_proba(X_test), name="predict_proba")
    df_pred_prob = pd.concat([df_test["psp_reference"], y_predict_proba], axis=1)
    df_pred_prob["created_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
    df_pred_prob["updated_at"] = pd.Series([datetime.datetime.now()] * df_pred_prob.shape[0])
    df_pred_prob.to_csv("../../predictions_dump_december.csv", index=False)
