"""
Tests for SQLAlchemy Database Operations, CRUD, and Case Updates.
Uses temporary in-memory database to test without modifying local database file.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.connection import Base
from src.database.crud import (
    insert_transaction_with_prediction,
    get_transactions_list,
    get_transaction_by_reference,
    update_investigation_record,
    get_dashboard_summary_metrics
)


@pytest.fixture
def test_db_session():
    """Creates a fresh in-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_insert_and_retrieve_transaction(test_db_session):
    """Verifies atomic insertion and retrieval of transaction, prediction, and case."""
    features = {"Time": 100.0, "Amount": 150.0, "V1": 0.5}
    txn = insert_transaction_with_prediction(
        db=test_db_session,
        transaction_ref="TXN-TEST-001",
        transaction_time=100.0,
        amount=150.0,
        features=features,
        fraud_probability=0.88,
        risk_score=88.0,
        risk_category="Critical Risk",
        predicted_label=1,
        anomaly_score=0.45
    )
    test_db_session.commit()

    assert txn.id is not None
    assert txn.transaction_reference == "TXN-TEST-001"

    # Query back
    retrieved = get_transaction_by_reference(test_db_session, "TXN-TEST-001")
    assert retrieved is not None
    assert retrieved["amount"] == 150.0
    assert retrieved["fraud_probability"] == 0.88
    assert retrieved["risk_category"] == "Critical Risk"
    assert retrieved["status"] == "New"


def test_update_investigation_case(test_db_session):
    """Verifies that an investigator can add notes and assign outcomes."""
    txn = insert_transaction_with_prediction(
        db=test_db_session,
        transaction_ref="TXN-TEST-002",
        transaction_time=200.0,
        amount=50.0,
        features={},
        fraud_probability=0.10,
        risk_score=10.0,
        risk_category="Low Risk",
        predicted_label=0
    )
    test_db_session.commit()

    # Update
    updated = update_investigation_record(
        db=test_db_session,
        transaction_id=txn.id,
        status="Under Review",
        notes="Cardholder reported traveling abroad.",
        outcome="Customer Verified"
    )
    test_db_session.commit()
    assert updated is True

    retrieved = get_transaction_by_reference(test_db_session, "TXN-TEST-002")
    assert retrieved["status"] == "Under Review"
    assert retrieved["notes"] == "Cardholder reported traveling abroad."
    assert retrieved["final_outcome"] == "Customer Verified"


def test_dashboard_metrics_aggregation(test_db_session):
    """Verifies aggregate count queries."""
    insert_transaction_with_prediction(
        db=test_db_session,
        transaction_ref="TXN-A",
        transaction_time=1.0,
        amount=10.0,
        features={},
        fraud_probability=0.1,
        risk_score=10.0,
        risk_category="Low Risk",
        predicted_label=0
    )
    insert_transaction_with_prediction(
        db=test_db_session,
        transaction_ref="TXN-B",
        transaction_time=2.0,
        amount=500.0,
        features={},
        fraud_probability=0.9,
        risk_score=90.0,
        risk_category="Critical Risk",
        predicted_label=1
    )
    test_db_session.commit()

    metrics = get_dashboard_summary_metrics(test_db_session)
    assert metrics["total_transactions"] == 2
    assert metrics["predicted_fraud_count"] == 1
    assert metrics["flagged_transactions"] == 1
