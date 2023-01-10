from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from src.common.schema import Filter


class CreatePrediction(BaseModel):
    predict_proba: float
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class ReadPrediction(CreatePrediction):
    class Config:
        orm_mode = True


class CreateTransaction(BaseModel):
    merchant: str
    card_schema: str
    is_credit: bool
    eur_amount: float
    ip_country: str
    issuing_country: str
    device_type: str
    ip_address: Optional[str] = None
    email_address: Optional[str] = None
    card_number: str
    shopper_interaction: str
    zip_code: Optional[str] = None
    card_bin: str
    has_fraudulent_dispute: bool
    is_refused_by_adyen: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class ReadTransaction(CreateTransaction):
    psp_reference: int
    predictions: Optional[ReadPrediction]

    class Config:
        orm_mode = True


class TransactionFilter(Filter):
    has_fraudulent_dispute: Optional[bool]
