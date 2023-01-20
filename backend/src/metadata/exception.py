from fastapi import HTTPException, status
from typing import Any, Dict, Optional

classifier_not_found_exception = HTTPException(
    status.HTTP_404_NOT_FOUND,
    detail="Classifier not found!",
)

explainer_not_found_exception = HTTPException(
    status.HTTP_404_NOT_FOUND,
    detail="Explainer not found!",
)

transaction_not_found_exception = HTTPException(
    status.HTTP_404_NOT_FOUND,
    detail="Transaction not found!",
)
