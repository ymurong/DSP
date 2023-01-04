from sqlalchemy.ext.declarative import declarative_base
from src.common.database_util import init_database

SQLALCHEMY_DATABASE_URL = 'sqlite:///./transactions.sqlite3'
engine, SessionLocal = init_database(SQLALCHEMY_DATABASE_URL)

# set up the orm
Base = declarative_base(bind=engine, name='Base')
