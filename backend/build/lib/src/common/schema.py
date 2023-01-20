from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class OrderBy(BaseModel):
    sort_by: Optional[str] = "created_at"
    is_desc: Optional[bool] = True


class Filter(BaseModel):
    created_at_from: Optional[datetime] = datetime(2020, 1, 1)
    created_at_to: Optional[datetime] = datetime.now()
