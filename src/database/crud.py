"""
Database CRUD (Create, Read, Update, Delete) Operations.
Provides parameterized, SQL-injection safe queries and aggregations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import desc, func, select, or_
from sqlalchemy.orm import Session

from src.database.models import Transaction, Prediction, Investigation
from src.database.connection import get_db_session
from src.utils.helpers import serialize_features, deserialize_features
from src.utils.logging_config import logger


def insert_transaction_with_prediction(
    db: Session,
    transaction_ref: str,
    transaction_time: float,
    amount: float,
    features: Dict[str, Any],
    fraud_probability: float,
    risk_score: float,
    risk_category: str,
    predicted_label: int,
    anomaly_score: Optional[float] = None,
    model_version: str = "v1.0.0"
) -> Transaction:
    """Inserts transaction, prediction, and associated investigation record atomically."""
    # Check if transaction reference exists
    existing = db.query(Transaction).filter(Transaction.transaction_reference == transaction_ref).first()
    if existing:
        return existing

    txn = Transaction(
        transaction_reference=transaction_ref,
        transaction_time=float(transaction_time),
        amount=float(amount),
        input_features=serialize_features(features)
    )
    db.add(txn)
    db.flush()  # Populates txn.id

    pred = Prediction(
        transaction_id=txn.id,
        fraud_probability=float(fraud_probability),
        risk_score=float(risk_score),
        risk_category=risk_category,
        predicted_label=int(predicted_label),
        anomaly_score=float(anomaly_score) if anomaly_score is not None else None,
        model_version=model_version
    )
    db.add(pred)

    inv = Investigation(
        transaction_id=txn.id,
        status="New",
        investigator_notes="",
        final_outcome="Pending"
    )
    db.add(inv)
    return txn


def get_transactions_list(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    risk_category: Optional[str] = None,
    min_risk_score: Optional[float] = None,
    investigation_status: Optional[str] = None,
    search_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves filtered transaction list with joined prediction and investigation data."""
    query = (
        db.query(Transaction, Prediction, Investigation)
        .join(Prediction, Transaction.id == Prediction.transaction_id)
        .outerjoin(Investigation, Transaction.id == Investigation.transaction_id)
    )

    if risk_category and risk_category != "All":
        query = query.filter(Prediction.risk_category == risk_category)

    if min_risk_score is not None and min_risk_score > 0:
        query = query.filter(Prediction.risk_score >= min_risk_score)

    if investigation_status and investigation_status != "All":
        query = query.filter(Investigation.status == investigation_status)

    if search_query:
        search_pattern = f"%{search_query.strip()}%"
        query = query.filter(Transaction.transaction_reference.ilike(search_pattern))

    query = query.order_by(desc(Transaction.id)).offset(offset).limit(limit)

    results = []
    for txn, pred, inv in query.all():
        results.append({
            "id": txn.id,
            "reference": txn.transaction_reference,
            "time": txn.transaction_time,
            "amount": txn.amount,
            "fraud_probability": pred.fraud_probability,
            "risk_score": pred.risk_score,
            "risk_category": pred.risk_category,
            "predicted_label": pred.predicted_label,
            "anomaly_score": pred.anomaly_score,
            "status": inv.status if inv else "New",
            "notes": inv.investigator_notes if inv else "",
            "final_outcome": inv.final_outcome if inv else "Pending",
            "created_at": txn.created_at.strftime("%Y-%m-%d %H:%M:%S") if txn.created_at else "",
            "features": deserialize_features(txn.input_features)
        })
    return results


def get_transaction_by_reference(db: Session, reference: str) -> Optional[Dict[str, Any]]:
    """Fetches single transaction details by reference code."""
    record = (
        db.query(Transaction, Prediction, Investigation)
        .join(Prediction, Transaction.id == Prediction.transaction_id)
        .outerjoin(Investigation, Transaction.id == Investigation.transaction_id)
        .filter(Transaction.transaction_reference == reference)
        .first()
    )
    if not record:
        return None

    txn, pred, inv = record
    return {
        "id": txn.id,
        "reference": txn.transaction_reference,
        "time": txn.transaction_time,
        "amount": txn.amount,
        "fraud_probability": pred.fraud_probability,
        "risk_score": pred.risk_score,
        "risk_category": pred.risk_category,
        "predicted_label": pred.predicted_label,
        "anomaly_score": pred.anomaly_score,
        "status": inv.status if inv else "New",
        "notes": inv.investigator_notes if inv else "",
        "final_outcome": inv.final_outcome if inv else "Pending",
        "created_at": txn.created_at.strftime("%Y-%m-%d %H:%M:%S") if txn.created_at else "",
        "features": deserialize_features(txn.input_features)
    }


def update_investigation_record(
    db: Session,
    transaction_id: int,
    status: str,
    notes: str,
    outcome: str
) -> bool:
    """Updates status, notes, and resolution outcome for an investigation."""
    inv = db.query(Investigation).filter(Investigation.transaction_id == transaction_id).first()
    if not inv:
        inv = Investigation(transaction_id=transaction_id)
        db.add(inv)

    inv.status = status
    inv.investigator_notes = notes
    inv.final_outcome = outcome
    db.flush()
    logger.info(f"Updated investigation for txn_id={transaction_id}: status={status}, outcome={outcome}")
    return True


def get_dashboard_summary_metrics(db: Session) -> Dict[str, Any]:
    """Calculates aggregate dashboard metrics."""
    total_txns = db.query(func.count(Transaction.id)).scalar() or 0
    if total_txns == 0:
        return {
            "total_transactions": 0,
            "flagged_transactions": 0,
            "predicted_fraud_count": 0,
            "high_risk_count": 0,
            "critical_risk_count": 0,
            "investigation_status_counts": {},
            "risk_category_counts": {}
        }

    predicted_fraud = db.query(func.count(Prediction.id)).filter(Prediction.predicted_label == 1).scalar() or 0
    high_risk = db.query(func.count(Prediction.id)).filter(Prediction.risk_category == "High Risk").scalar() or 0
    critical_risk = db.query(func.count(Prediction.id)).filter(Prediction.risk_category == "Critical Risk").scalar() or 0
    flagged = high_risk + critical_risk

    # Group by risk category
    risk_groups = (
        db.query(Prediction.risk_category, func.count(Prediction.id))
        .group_by(Prediction.risk_category)
        .all()
    )
    risk_dist = {cat: count for cat, count in risk_groups}

    # Group by investigation status
    inv_groups = (
        db.query(Investigation.status, func.count(Investigation.id))
        .group_by(Investigation.status)
        .all()
    )
    inv_dist = {st: count for st, count in inv_groups}

    return {
        "total_transactions": total_txns,
        "flagged_transactions": flagged,
        "predicted_fraud_count": predicted_fraud,
        "high_risk_count": high_risk,
        "critical_risk_count": critical_risk,
        "investigation_status_counts": inv_dist,
        "risk_category_counts": risk_dist
    }


def get_recent_alerts(db: Session, limit: int = 15) -> List[Dict[str, Any]]:
    """Fetches highest risk alerts needing investigator attention."""
    query = (
        db.query(Transaction, Prediction, Investigation)
        .join(Prediction, Transaction.id == Prediction.transaction_id)
        .outerjoin(Investigation, Transaction.id == Investigation.transaction_id)
        .filter(or_(Prediction.risk_category == "High Risk", Prediction.risk_category == "Critical Risk"))
        .order_by(desc(Prediction.risk_score), desc(Transaction.id))
        .limit(limit)
    )

    alerts = []
    for txn, pred, inv in query.all():
        alerts.append({
            "reference": txn.transaction_reference,
            "risk_score": pred.risk_score,
            "risk_category": pred.risk_category,
            "fraud_probability": pred.fraud_probability,
            "amount": txn.amount,
            "status": inv.status if inv else "New",
            "created_at": txn.created_at.strftime("%Y-%m-%d %H:%M:%S") if txn.created_at else ""
        })
    return alerts
