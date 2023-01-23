from pydantic import BaseModel
from typing import Dict, Optional


class FairnessMetrics(BaseModel):
    balanced_accuracy: Dict[str, float]
    false_positive_rate: Dict[str, float]
    false_negative_rate: Dict[str, float]


class ClassifierMetrics(BaseModel):
    f1: float
    recall: float
    precision: float
    auc: float
    block_rate: float
    fraud_rate: float
    fairness: Optional[FairnessMetrics]


class StoreMetrics(BaseModel):
    merchant: str
    total_revenue: float
    chargeback_costs: float


class ExplainabilityScore(BaseModel):
    ip_risk: float
    email_risk: float
    risk_card_behaviour: float
    risk_card_amount: float
    general_evidences: float


class MonthMetrics(BaseModel):
    month: str
    block_rate: float
    fraud_rate: float
    total_revenue: float
    chargeback_costs: float
