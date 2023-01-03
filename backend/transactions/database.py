from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = 'sqlite:///./transactions.sqlite3'

engine = create_engine(
    # check_same_thread = false enables sqlalchemy multi-threading feature
    SQLALCHEMY_DATABASE_URL, encoding='utf-8', echo=True, connect_args={'check_same_thread': False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=True)

Base = declarative_base(bind=engine, name='Base')