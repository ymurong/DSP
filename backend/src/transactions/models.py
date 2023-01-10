from sqlalchemy import Column, String, BigInteger, Boolean, Float, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship, backref
from random import randint

from .database import Base, engine


def random_integer():
    min_ = 10000000000
    max_ = 99999999999
    rand = randint(min_, max_)

    # possibility of same random number is very low.
    # but if you want to make sure, here you can check id exists in database.
    from sqlalchemy.orm import sessionmaker
    db_session_maker = sessionmaker(bind=engine)
    db_session = db_session_maker()
    while db_session.query(Transactions).filter(Transactions.psp_reference == rand).limit(1).first() is not None:
        rand = randint(min_, max_)

    return rand


class Transactions(Base):
    __tablename__ = 'transactions'  # 数据表的表名

    psp_reference = Column(BigInteger, primary_key=True, index=True, default=random_integer)
    merchant = Column(String(1), nullable=False, comment='merchant type')
    card_schema = Column(String(10), nullable=False, comment='card type')
    is_credit = Column(Boolean, nullable=False, comment='is credit card or not')
    eur_amount = Column(Float, nullable=False)
    ip_country = Column(String(2), nullable=False)
    issuing_country = Column(String(2), nullable=False)
    device_type = Column(String(10), nullable=False)
    ip_address = Column(String(20), nullable=True)
    email_address = Column(String(20), nullable=True)
    card_number = Column(String(20), nullable=True)
    shopper_interaction = Column(String(20), nullable=True)
    zip_code = Column(String(10), nullable=True)
    card_bin = Column(String(4), nullable=False)
    has_fraudulent_dispute = Column(Boolean, nullable=True)
    is_refused_by_adyen = Column(Boolean, nullable=False)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    predictions = relationship("Predictions", back_populates="transactions", uselist=False)

    def __repr__(self):
        return f'{self.psp_reference}_{self.created_at}'


class Predictions(Base):
    __tablename__ = 'predictions'  # 数据表的表名

    psp_reference = Column(BigInteger, ForeignKey("transactions.psp_reference"), primary_key=True, index=True)
    predict_proba = Column(Float, nullable=False, comment='prediction probability')
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    transactions = relationship("Transactions", back_populates="predictions")

    def __repr__(self):
        return f'{self.psp_reference}: predict_proba={self.predict_proba}'
