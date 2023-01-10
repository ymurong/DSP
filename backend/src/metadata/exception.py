from fastapi import HTTPException, status

classifier_not_found_exception = HTTPException(
    status.HTTP_404_NOT_FOUND,
    detail="Classifier not found!",
)
