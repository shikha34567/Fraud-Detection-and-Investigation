"""
Tests for Model Inference, Output Shapes, and Anomaly Scores.
"""

import pytest
import pandas as pd
from src.models.predict import FraudPredictor
from src.utils.helpers import generate_synthetic_transactions


def test_predictor_loaded_and_predicts():
    """Verifies FraudPredictor loads persisted artifacts and yields valid predictions."""
    predictor = FraudPredictor()
    loaded = predictor.load_artifacts()
    assert loaded is True
    assert predictor.is_loaded is True

    # Test single transaction prediction
    df = generate_synthetic_transactions(n_samples=5)
    sample_txn = df.iloc[0].drop("Class").to_dict()
    result = predictor.predict_single(sample_txn)

    assert "fraud_probability" in result
    assert "risk_score" in result
    assert "risk_category" in result
    assert "anomaly_score" in result

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert 0.0 <= result["risk_score"] <= 100.0
    assert result["risk_category"] in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]


def test_batch_prediction_shape():
    """Verifies batch DataFrame predictions maintain shape and add scoring columns."""
    predictor = FraudPredictor()
    predictor.load_artifacts()

    df = generate_synthetic_transactions(n_samples=15)
    features_df = df.drop(columns=["Class"])
    scored_df = predictor.predict_dataframe(features_df)

    assert len(scored_df) == 15
    assert "fraud_probability" in scored_df.columns
    assert "risk_score" in scored_df.columns
    assert "risk_category" in scored_df.columns
    assert "anomaly_score" in scored_df.columns
