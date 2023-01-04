from sqlalchemy.orm import Session
from . import models, schemas
from fastapi_pagination.ext.sqlalchemy import paginate
from src.common.query_util import query_filter, query_sort
from src.common.schema import OrderBy


def get_transactions(db: Session, filters: schemas.TransactionFilter, order_by: OrderBy):
    queryset = db.query(models.Transactions)
    queryset = query_filter(queryset, filters, models.Transactions)
    queryset = query_sort(queryset, order_by, models.Transactions)
    return paginate(queryset)


def create_transaction(db: Session, transaction: schemas.CreateTransaction):
    db_transactions = models.Transactions(**transaction.dict())
    db.add(db_transactions)
    db.commit()
    db.refresh(db_transactions)
    return db_transactions
