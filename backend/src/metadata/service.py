from src.common.RFClassifierPipeline import RFClassifierPipeline
from src.common.XGBClassifierPipeline import XGBClassifierPipeline
from src.common.BasePipeline import BasePipeline
import pandas as pd
import os
from sqlalchemy.orm import Session
from src.transactions import models
from sqlalchemy import func
from src.metadata.exception import classifier_not_found_exception, explainer_not_found_exception, \
    transaction_not_found_exception
from src.resources.conf import INPUT_FEATURES, OUTPUT_FEATURE, TEST_DATA_PATH, XGBOOST_MODEL_PATH, RF_MODEL_PATH, \
    RF_EXPLAINER_PATH, SENSITIVE_FEATURE

import logging

logger = logging.getLogger("transactions_api")


def load_test_data_by_psp_ref(psp_reference: int):
    dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(dir, TEST_DATA_PATH)
    df_test = pd.read_csv(fname)
    df_test = df_test[df_test["psp_reference"] == psp_reference]
    if df_test.shape[0] == 0:
        logger.error(f"transaction {psp_reference} not found in local testing data")
        raise transaction_not_found_exception
    X_test = df_test[INPUT_FEATURES]
    y_test = df_test[OUTPUT_FEATURE]
    return X_test, y_test


def load_test_data(sensitive_features_included=False):
    dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(dir, TEST_DATA_PATH)
    df_test = pd.read_csv(fname)
    X_test = df_test[INPUT_FEATURES]
    y_test = df_test[OUTPUT_FEATURE]
    if sensitive_features_included:
        A_test = df_test[SENSITIVE_FEATURE]
        return X_test, y_test, A_test
    return X_test, y_test


def classifier_factory(classifier_name: str) -> BasePipeline:
    if classifier_name == "xgboost":
        return XGBClassifierPipeline(model_file_name=XGBOOST_MODEL_PATH)
    if classifier_name == "random_forest":
        return RFClassifierPipeline(model_file_name=RF_MODEL_PATH)
    raise classifier_not_found_exception


def explainer_factory(explainer_name: str) -> BasePipeline:
    if explainer_name == "random_forest_lime":
        rf0 = RFClassifierPipeline(model_file_name=RF_MODEL_PATH)
        rf0.load_explainer(RF_EXPLAINER_PATH)
        return rf0
    raise explainer_not_found_exception


def get_explainability_scores(psp_reference: int, explainer_name: str) -> dict:
    X_test, _ = load_test_data_by_psp_ref(psp_reference)
    transaction_sample = X_test.values[0]
    pipeline = explainer_factory(explainer_name=explainer_name)
    explanability_scores = pipeline.explain(transaction_sample)
    return explanability_scores

def get_explainability_features(psp_reference: int, explainer_name: str) -> list:
    X_test, _ = load_test_data_by_psp_ref(psp_reference)
    transaction_sample = X_test.values[0]
    pipeline = explainer_factory(explainer_name=explainer_name)
    explanability_features = pipeline.get_influential_features(transaction_sample)
    return explanability_features
                                 


def get_classifier_metrics(classifier_name: str, threshold: float = 0.5) -> dict:
    X_test, y_test, A_test = load_test_data(sensitive_features_included=True)
    pipeline = classifier_factory(classifier_name)
    metrics = pipeline.eval(X_test, y_test, threshold=threshold)
    fairness_metrics = pipeline.eval_fairness(X_test, y_test, A_test, threshold=threshold)
    return {
        **metrics,
        "fairness": {**fairness_metrics}
    }


def get_store_metrics(db: Session, threshold: float) -> dict:
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


def get_month_metrics(db: Session, threshold: float):
    response = []
    for month in range(1, 12 + 1):
        response.append(get_metrics_by_month(db, threshold, month))
    return response


def get_metrics_by_month(db: Session, threshold: float, month: int):
    month_from, month_to, year_from, year_to = get_month_ranges(month=month, year=2021)

    total_transactions = db.query(
        func.count(models.Transactions.psp_reference)
    ). \
        filter(models.Transactions.created_at.between(str(year_from) + "-" + str(month_from) + "-01",
                                                      str(year_to) + "-" + str(month_to) + "-01")).all()

    fraud_transactions = db.query(
        func.count(models.Transactions.psp_reference)
    ). \
        join(models.Predictions). \
        filter(models.Transactions.created_at.between(str(year_from) + "-" + str(month_from) + "-01",
                                                      str(year_to) + "-" + str(month_to) + "-01")). \
        filter(models.Transactions.has_fraudulent_dispute == True). \
        filter(models.Predictions.predict_proba < threshold).all()

    block_transactions = db.query(
        func.count(models.Transactions.psp_reference)
    ). \
        join(models.Predictions). \
        filter(models.Transactions.created_at.between(str(year_from) + "-" + str(month_from) + "-01",
                                                      str(year_to) + "-" + str(month_to) + "-01")). \
        filter(models.Predictions.predict_proba > threshold).all()

    total_revenue = db.query(
        func.sum(models.Transactions.eur_amount),
    ). \
        join(models.Predictions). \
        filter(models.Transactions.created_at.between(str(year_from) + "-" + str(month_from) + "-01",
                                                      str(year_to) + "-" + str(month_to) + "-01")). \
        filter(models.Predictions.predict_proba < threshold).all()

    chargeback_costs = db.query(
        func.count(models.Transactions.psp_reference) * 15 + func.sum(models.Transactions.eur_amount),
    ). \
        join(models.Predictions). \
        filter(models.Transactions.created_at.between(str(year_from) + "-" + str(month_from) + "-01",
                                                      str(year_to) + "-" + str(month_to) + "-01")). \
        filter(models.Transactions.has_fraudulent_dispute == True). \
        filter(models.Predictions.predict_proba < threshold).all()

    chargeback_costs = chargeback_costs[0][0] if chargeback_costs[0][0] != None else 0
    total_revenue = total_revenue[0][0] if total_revenue[0][0] != None else 0

    dict_month_analytics = {
        "month": str(month),
        "block_rate": block_transactions[0][0] / total_transactions[0][0],
        "fraud_rate": fraud_transactions[0][0] / total_transactions[0][0],
        "total_revenue": total_revenue,
        "chargeback_costs": chargeback_costs,
    }
    return dict_month_analytics


def get_month_ranges(month: int, year: int):
    month_from = month
    year_from = year
    month_to = month + 1
    year_to = year
    if month_to > 12:
        month_to = 1
        year_to = year + 1

    return "{:02d}".format(month_from), "{:02d}".format(month_to), str(year_from), str(year_to)
