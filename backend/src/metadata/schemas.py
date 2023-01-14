from pydantic import BaseModel


class ClassifierMetrics(BaseModel):
    f1: float
    recall: float
    precision: float
    auc: float
    block_rate: float
    fraud_rate: float


class StoreMetrics(BaseModel):
    merchant: str
    total_revenue: float
    chargeback_costs: float
