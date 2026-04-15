import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, DECIMAL, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5433/fraud_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(String(255), nullable=False)
    amount = Column(DECIMAL(15, 2), nullable=False)
    merchant_category = Column(String(255))
    location = Column(String(255))
    timestamp = Column(DateTime, nullable=False)
    status = Column(String(50), default='PROCESSED')
    risk_score = Column(Integer, default=0)
    is_fraud = Column(Boolean, nullable=True, default=None)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(255), nullable=True)
    rule_triggered = Column(String(255), nullable=False)
    severity = Column(String(50), nullable=False)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(255), nullable=False)
    entity_id = Column(String(255))
    details = Column(Text)
    timestamp = Column(DateTime)
