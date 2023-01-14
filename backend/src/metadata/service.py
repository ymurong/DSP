from src.common.XGBClassifierPipeline import XGBClassifierPipeline
from src.common.BasePipeline import BasePipeline
import pandas as pd
from sqlalchemy.orm import Session
from src.transactions import models
from sqlalchemy import func
from src.metadata.exception import classifier_not_found_exception

TEST_DATA_PATH = "src/resources/test_dataset_december.csv"
XGBOOST_MODEL_PATH = "../resources/pretrained_models/xgboost_classifier_model.pkl"
FEATURES = ['ip_node_degree', 'card_node_degree', 'email_node_degree', 'is_credit',
            'ip_address_woe', 'email_address_woe', 'card_number_woe', 'no_ip',
            'no_email', 'same_country', 'merchant_Merchant B',
            'merchant_Merchant C', 'merchant_Merchant D', 'merchant_Merchant E',
            'card_scheme_MasterCard', 'card_scheme_Other', 'card_scheme_Visa',
            'device_type_Linux', 'device_type_MacOS', 'device_type_Other',
            'device_type_Windows', 'device_type_iOS', 'shopper_interaction_POS']


def load_test_data():
    df_test = pd.read_csv(TEST_DATA_PATH)
    X_test = df_test[FEATURES]
    y_test = df_test["has_fraudulent_dispute"]
    return X_test, y_test


def classifier_factory(classifier_name: str) -> BasePipeline:
    if classifier_name == "xgboost":
        return XGBClassifierPipeline(model_file_name=XGBOOST_MODEL_PATH)
    raise classifier_not_found_exception


def get_classifier_metrics(classifier_name: str = "xgboost", threshold: float = 0.5):
    X_test, y_test = load_test_data()
    pipeline = classifier_factory(classifier_name)
    metrics = pipeline.eval(X_test, y_test, threshold=threshold)
    return metrics


def get_store_metrics(db: Session, threshold: float):
    """
    chargeback_costs: value of transaction + 15 standard fee
    total_revenue: total amount of approved volume processed by the merchant minus the chargeback costs
    """

    chargeback_costs = db.query(
        models.Transactions.merchant,
        func.count(models.Transactions.psp_reference) * 15 + func.sum(models.Transactions.eur_amount),
    ). \
        join(models.Predictions). \
        filter(models.Transactions.has_fraudulent_dispute == True). \
        filter(models.Predictions.predict_proba < threshold). \
        group_by(models.Transactions.merchant).all()
    chargeback_costs = dict(chargeback_costs)

    total_revenue = db.query(
        models.Transactions.merchant,
        func.sum(models.Transactions.eur_amount),
    ). \
        join(models.Predictions). \
        filter(models.Predictions.predict_proba < threshold). \
        group_by(models.Transactions.merchant).all()
    total_revenue = dict(total_revenue)

    # organize final store metrics for each merchant
    store_metrics = []
    for k in total_revenue.keys():
        metrics = {
            "merchant": k,
            "chargeback_costs": round(chargeback_costs[k], 2),
            "total_revenue": round(total_revenue[k] - chargeback_costs[k], 2)
        }
        store_metrics.append(metrics)
    return store_metrics
