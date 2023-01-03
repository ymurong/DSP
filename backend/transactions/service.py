from sqlalchemy.orm import Session
from . import models, schemas


def get_transactions(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Transactions).offset(skip).limit(limit).all()


def create_transaction(db: Session, transaction: schemas.CreateTransaction):
    db_transactions = models.Transactions(**transaction.dict())
    db.add(db_transactions)
    db.commit()
    db.refresh(db_transactions)
    return db_transactions
