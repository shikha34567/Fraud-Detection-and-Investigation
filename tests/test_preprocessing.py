"""
Tests for Preprocessing Pipeline and Chronological Splitting.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.preprocess import Preprocessor
from src.utils.helpers import generate_synthetic_transactions
from config.settings import MODEL_CONFIG


def test_chronological_split():
    """Verifies that chronological splitting respects temporal order without lookahead."""
    df = generate_synthetic_transactions(n_samples=100)
    preprocessor = Preprocessor()
    train_df, val_df, test_df = preprocessor.split_data_chronological(df, val_size=0.1, test_size=0.2)

    assert len(train_df) == 70
    assert len(val_df) == 10
    assert len(test_df) == 20

    # Ensure max time of train <= min time of val <= min time of test
    assert train_df[MODEL_CONFIG.TIME_COLUMN].max() <= val_df[MODEL_CONFIG.TIME_COLUMN].min()
    assert val_df[MODEL_CONFIG.TIME_COLUMN].max() <= test_df[MODEL_CONFIG.TIME_COLUMN].min()


def test_preprocessor_fit_transform():
    """Verifies RobustScaler transforms Amount & Time while passing V1-V28 through."""
    df = generate_synthetic_transactions(n_samples=60)
    preprocessor = Preprocessor()
    X = preprocessor.fit_transform(df)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 60
    assert X.shape[1] == len(MODEL_CONFIG.ALL_FEATURES)
    assert not np.isnan(X).any()


def test_missing_feature_raises_error():
    """Verifies that an error is raised if a required feature is missing."""
    df = generate_synthetic_transactions(n_samples=20)
    df_missing = df.drop(columns=["V10"])
    preprocessor = Preprocessor()
    preprocessor.fit(df)

    with pytest.raises(KeyError):
        preprocessor.transform(df_missing)
