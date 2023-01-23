from fastapi import APIRouter, Query, Depends, Path
from . import service, schemas
from src.transactions.database import engine, Base, SessionLocal
from sqlalchemy.orm import Session
from typing import List
from src.metadata.enum import ClassifierEnum

metadata_app = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@metadata_app.get("/classifier/metrics/{classifier_name}", response_model=schemas.ClassifierMetrics,
                  description="Provide chosen pretrained classifier metrics such as f1, precision, recall, auc, block_rate, fraud_rate")
def get_classifier_metrics(
        classifier_name: ClassifierEnum,
        threshold: float = Query(0.5, ge=0, le=1),
):
    classifier_metrics = service.get_classifier_metrics(classifier_name=classifier_name, threshold=threshold)
    return classifier_metrics


@metadata_app.get("/store/metrics", response_model=List[schemas.StoreMetrics],
                  description="Provide store metrics such as total_revenue, chargeback_costs")
def get_store_metrics(
        db: Session = Depends(get_db),
        threshold: float = Query(0.5, ge=0, le=1),
):
    store_metrics = service.get_store_metrics(db, threshold)
    return store_metrics

@metadata_app.get("/classifier/metrics/monthly", response_model=List[schemas.MonthMetrics],
                  description="Provide analytic metrics per month")
def get_month_metrics(
        db: Session = Depends(get_db),
        threshold: float = Query(0.5, ge=0, le=1),
):
   month_metrics = service.get_month_metrics(db, threshold)
   return month_metrics