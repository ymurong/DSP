from fastapi import APIRouter, Depends
from .database import engine, Base, SessionLocal
from . import service, schemas
from sqlalchemy.orm import Session
from fastapi_pagination import Page
from src.common.schema import OrderBy

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


@transaction_app.post("", response_model=schemas.ReadTransaction)
def create_transaction(transaction: schemas.CreateTransaction, db: Session = Depends(get_db)):
    return service.create_transaction(db=db, transaction=transaction)
