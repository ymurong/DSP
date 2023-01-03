from fastapi import APIRouter, Depends
from .database import engine, Base, SessionLocal
from . import service, schemas
from typing import List
from sqlalchemy.orm import Session

transaction_app = APIRouter()
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@transaction_app.get("/", response_model=List[schemas.ReadTransaction])
def get_transactions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    transactions = service.get_transactions(db, skip=skip, limit=limit)
    return transactions


@transaction_app.post("/", response_model=schemas.ReadTransaction)
def create_transaction(transaction: schemas.CreateTransaction, db: Session = Depends(get_db)):
    return service.create_transaction(db=db, transaction=transaction)
