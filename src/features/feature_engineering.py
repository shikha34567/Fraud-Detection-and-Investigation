"""
Feature Engineering module for financial transaction streams.
Extracts cyclical time indicators, normalized transaction magnitudes, and interaction ratios.
"""

from typing import List
import numpy as np
import pandas as pd
from config.settings import MODEL_CONFIG


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes helpful domain-specific indicators while retaining base schema compatibility.
    - hour_of_day: Periodicity indicator (0 - 23)
    - log_amount: Log-transformed transaction amount handling long tails
    - high_amount_flag: Binary indicator for unusually large transactions (> $1,000)
    """
    df_out = df.copy()

    if MODEL_CONFIG.TIME_COLUMN in df_out.columns:
        # Time represents seconds elapsed; convert to hour of day (0-23)
        hours = (df_out[MODEL_CONFIG.TIME_COLUMN] // 3600) % 24
        df_out["hour_of_day"] = hours.astype(int)

    if MODEL_CONFIG.AMOUNT_COLUMN in df_out.columns:
        df_out["log_amount"] = np.log1p(np.maximum(df_out[MODEL_CONFIG.AMOUNT_COLUMN], 0))
        df_out["high_amount_flag"] = (df_out[MODEL_CONFIG.AMOUNT_COLUMN] >= 1000.0).astype(int)

    return df_out


def get_feature_names() -> List[str]:
    """Returns list of all raw model features."""
    return MODEL_CONFIG.ALL_FEATURES
