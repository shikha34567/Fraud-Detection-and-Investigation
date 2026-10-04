"""
Tests for Risk Scoring Logic and Threshold Classifications.
"""

import pytest
import numpy as np
from src.models.risk_scoring import RiskScorer


def test_risk_score_bounds():
    """Verifies risk score remains strictly bounded between 0.0 and 100.0."""
    scorer = RiskScorer()
    assert scorer.compute_risk_score(0.0) == 0.0
    assert scorer.compute_risk_score(1.0) == 100.0
    assert scorer.compute_risk_score(0.4567) == 45.67

    # Edge cases
    assert scorer.compute_risk_score(-0.2) == 0.0
    assert scorer.compute_risk_score(1.5) == 100.0

    # Array input
    arr = np.array([0.0, 0.5, 1.0])
    scores = scorer.compute_risk_score(arr)
    assert np.allclose(scores, [0.0, 50.0, 100.0])


def test_risk_category_assignment():
    """Verifies qualitative risk categories match defined thresholds."""
    scorer = RiskScorer(low_upper=25.0, medium_upper=60.0, high_upper=85.0)

    assert scorer.assign_risk_category(10.0) == "Low Risk"
    assert scorer.assign_risk_category(24.9) == "Low Risk"
    assert scorer.assign_risk_category(25.0) == "Medium Risk"
    assert scorer.assign_risk_category(59.9) == "Medium Risk"
    assert scorer.assign_risk_category(60.0) == "High Risk"
    assert scorer.assign_risk_category(84.9) == "High Risk"
    assert scorer.assign_risk_category(85.0) == "Critical Risk"
    assert scorer.assign_risk_category(99.9) == "Critical Risk"


def test_score_transaction_payload():
    """Verifies structured scoring dictionary."""
    scorer = RiskScorer()
    res = scorer.score_transaction(fraud_probability=0.72, anomaly_score=1.34)
    assert res["fraud_probability"] == 0.72
    assert res["risk_score"] == 72.0
    assert res["risk_category"] == "High Risk"
    assert res["is_flagged"] is True
    assert res["anomaly_score"] == 1.34
