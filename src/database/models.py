"""
SQLAlchemy ORM Data Models for Transactions, Risk Predictions, and Investigations.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text
)
from sqlalchemy.orm import relationship
from src.database.connection import Base


def utcnow() -> datetime:
    """Returns current UTC timestamp."""
    return datetime.now(timezone.utc)


class Transaction(Base):
    """Stores incoming financial transactions."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    transaction_reference = Column(String(64), unique=True, nullable=False, index=True)
    transaction_time = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    input_features = Column(Text, nullable=False)  # JSON serialized features (V1..V28, etc.)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    prediction = relationship("Prediction", back_populates="transaction", uselist=False, cascade="all, delete-orphan")
    investigation = relationship("Investigation", back_populates="transaction", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Transaction ref={self.transaction_reference} amount=${self.amount:.2f}>"


class Prediction(Base):
    """Stores ML model inference outputs and risk scores."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="CASCADE"), unique=True, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False, index=True)
    risk_category = Column(String(32), nullable=False, index=True)  # Low Risk, Medium Risk, High Risk, Critical Risk
    predicted_label = Column(Integer, nullable=False)  # 0 or 1
    anomaly_score = Column(Float, nullable=True)
    model_version = Column(String(64), default="v1.0.0", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    transaction = relationship("Transaction", back_populates="prediction")

    def __repr__(self) -> str:
        return f"<Prediction txn_id={self.transaction_id} score={self.risk_score} cat={self.risk_category}>"


class Investigation(Base):
    """Stores investigator review history, notes, and resolution outcomes."""
    __tablename__ = "investigations"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id", ondelete="CASCADE"), unique=True, nullable=False)
    status = Column(String(32), default="New", nullable=False, index=True)  # New, Under Review, Confirmed Fraud, Legitimate, Escalated
    investigator_notes = Column(Text, default="", nullable=True)
    final_outcome = Column(String(64), default="Pending", nullable=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    transaction = relationship("Transaction", back_populates="investigation")

    def __repr__(self) -> str:
        return f"<Investigation txn_id={self.transaction_id} status={self.status}>"
