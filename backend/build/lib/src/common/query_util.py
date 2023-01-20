from sqlalchemy.orm import Query
from pydantic import BaseModel
from sqlalchemy.orm.decl_api import DeclarativeMeta
from .schema import OrderBy
from datetime import datetime


def query_filter(queryset: Query, filters: BaseModel, model: DeclarativeMeta):
    """
    :param queryset: sqlalchemy intermediate query
    :param filters: pydantic schema filter
    :param model: sqlalchemy DeclarativeMeta (database model)
    :return: queryset: updated Query
    """
    filters_dict = filters.dict()
    for attr in [x for x in filters_dict if filters_dict[x] is not None]:
        if filters.__fields__[attr].type_ is bool:
            queryset = queryset.filter(getattr(model, attr).is_(filters_dict[attr]))
        elif filters.__fields__[attr].type_ is str:
            queryset = queryset.filter(getattr(model, attr).like(filters_dict[attr]))

    # specific handling of created_at time span filtering
    if filters_dict["created_at_from"] is not None:
        queryset = queryset.filter(getattr(model, "created_at") >= filters_dict["created_at_from"])
    if filters_dict["created_at_to"] is not None:
        queryset = queryset.filter(getattr(model, "created_at") <= filters_dict["created_at_to"])
    return queryset


def query_sort(queryset: Query, order_by: OrderBy, model: DeclarativeMeta):
    """
    :param queryset: sqlalchemy intermediate query
    :param order_by: pydantic OrderBy schema
    :param model: sqlalchemy DeclarativeMeta (database model)
    :return: queryset: updated Query
    """
    if order_by.sort_by is not None and order_by.is_desc is not None:
        queryset = queryset.order_by(getattr(getattr(model, order_by.sort_by), "desc" if order_by.is_desc else "asc")())
    return queryset
