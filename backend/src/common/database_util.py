from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def init_database(sqlalchemy_database_url):
    engine = create_engine(
        # check_same_thread = false enables sqlalchemy multi-threading feature
        sqlalchemy_database_url, encoding='utf-8', echo=True, connect_args={'check_same_thread': False}
    )
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=True)
    return engine, session_local
