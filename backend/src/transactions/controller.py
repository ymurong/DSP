from fastapi import APIRouter, Depends, Query
from .database import engine, Base, SessionLocal
from . import service, schemas
from src.metadata import service as metadata_service
from sqlalchemy.orm import Session
from fastapi_pagination import Page
from src.common.schema import OrderBy
from .enum import ExplainerEnum
from src.metadata.schemas import ExplainabilityScore

transaction_app = APIRouter()
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@transaction_app.get("", response_model=Page[schemas.ReadTransaction],
                     description="Provide transactions with predicted records")
def get_transactions(db: Session = Depends(get_db), filters: schemas.TransactionFilter = Depends(),
                     order_by: OrderBy = Depends()):
    transactions = service.get_predicted_transactions(db, filters, order_by)
    return transactions


@transaction_app.get("/{psp_reference}/explainability_score", response_model=ExplainabilityScore,
                     description="Explain transaction based on predefined category")
def explain_transaction(
        psp_reference: int,
        explainer_name: ExplainerEnum = Query(ExplainerEnum.random_forest_lime)
):
    explainability_scores = metadata_service.get_explainability_scores(psp_reference=psp_reference, explainer_name=explainer_name)
    return explainability_scores

@transaction_app.get("/{psp_reference}/explainability_features", response_model=list,
                     description="Get most influential features")
def get_influential_features(
        psp_reference: int,
        explainer_name: ExplainerEnum = Query(ExplainerEnum.random_forest_lime)
):
    explainability_features = metadata_service.get_explainability_features(psp_reference=psp_reference, explainer_name=explainer_name)
    return explainability_features

@transaction_app.post("", response_model=schemas.ReadTransaction)
def create_transaction(transaction: schemas.CreateTransaction, db: Session = Depends(get_db)):
    return service.create_transaction(db=db, transaction=transaction)
