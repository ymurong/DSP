import pytest
from src.common.database_util import init_database
from ..controller import get_db
from fastapi.testclient import TestClient
from src.run import app
import re
from ..database import Base

# using a test database in pytest based on https://fastapi.tiangolo.com/advanced/testing-database/
SQLALCHEMY_DATABASE_URL = 'sqlite:///./test_transactions.sqlite3'
engine, TestingSessionLocal = init_database(SQLALCHEMY_DATABASE_URL)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture()
def test_db():
    """
    Before each test, create all tables based on the same ORM definition
    After each test, drop all tables
    :return:
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# override get_db function in order to use testing database dependency
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture()
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_create_transaction(test_db):
    """
    - create one fake transaction
    - post call response should be 200 with 11-digit number psp_reference
    """
    response = client.post("/transactions", json={
        "merchant": "string",
        "card_schema": "string",
        "is_credit": True,
        "eur_amount": 0,
        "ip_country": "string",
        "issuing_country": "string",
        "device_type": "string",
        "ip_address": "string",
        "email_address": "string",
        "card_number": "string",
        "shopper_interaction": "string",
        "zip_code": "string",
        "card_bin": "string",
        "has_fraudulent_dispute": True,
        "is_refused_by_adyen": True,
        "created_at": "2023-01-04T12:34:52.459Z",
        "updated_at": "2023-01-04T12:34:52.459Z"
    })
    assert response.status_code == 200
    assert re.match("^[0-9]{11}$", str(response.json()["psp_reference"]))


def test_get_transactions(test_db):
    """
    - inject one fake transaction
    - get call response should be 200 with only 1 transaction in items
    """
    client.post("/transactions", json={
        "merchant": "fake",
        "card_schema": "fake",
        "is_credit": True,
        "eur_amount": 0,
        "ip_country": "fake",
        "issuing_country": "fake",
        "device_type": "fake",
        "ip_address": "fake",
        "email_address": "fake",
        "card_number": "fake",
        "shopper_interaction": "fake",
        "zip_code": "fake",
        "card_bin": "fake",
        "has_fraudulent_dispute": True,
        "is_refused_by_adyen": True,
        "created_at": "2023-01-04T12:34:52.459Z",
        "updated_at": "2023-01-04T12:34:52.459Z"
    })
    response = client.get("/transactions?page=1&size=50")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1

