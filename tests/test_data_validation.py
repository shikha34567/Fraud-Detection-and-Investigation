"""
Tests for Data Validation and Quality Auditing.
"""

import pytest
import pandas as pd
from src.data.load_data import load_dataset
from src.data.validate_data import DataValidator
from src.utils.helpers import generate_synthetic_transactions


def test_synthetic_data_schema():
    """Verifies generated sample data complies with ULB feature schema."""
    df = generate_synthetic_transactions(n_samples=50, fraud_ratio=0.1)
    validator = DataValidator()
    res = validator.validate_schema(df)
    assert res["is_valid"] is True
    assert len(res["missing_columns"]) == 0
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "Time" in df.columns
    assert "V1" in df.columns and "V28" in df.columns


def test_missing_and_duplicate_detection():
    """Verifies that validator accurately catches introduced nulls and duplicates."""
    df = generate_synthetic_transactions(n_samples=40)
    # Introduce null
    df.loc[0, "Amount"] = None
    # Duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    validator = DataValidator()
    report = validator.generate_quality_report(df, is_demo=True)
    assert report["missing_values"]["total_missing"] >= 1
    assert report["duplicate_records"] >= 1
    assert len(report["validation_warnings"]) >= 1


def test_load_dataset_fallback():
    """Verifies DataLoader fallback mechanism works when raw data file is absent."""
    df, is_demo = load_dataset(use_demo_if_missing=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Class" in df.columns
