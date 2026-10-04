"""
Utility functions for transaction processing, ID generation, and data conversion.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict
import numpy as np
import pandas as pd


def generate_transaction_reference() -> str:
    """Generates a unique, standardized transaction reference identifier."""
    unique_suffix = uuid.uuid4().hex[:8].upper()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    return f"TXN-{timestamp}-{unique_suffix}"


def serialize_features(features: Dict[str, Any] | pd.Series) -> str:
    """Serializes feature dictionary to JSON string."""
    if isinstance(features, pd.Series):
        features = features.to_dict()

    # Convert numpy types to python native types
    cleaned = {}
    for k, v in features.items():
        if isinstance(v, (np.floating, float)):
            cleaned[k] = round(float(v), 5)
        elif isinstance(v, (np.integer, int)):
            cleaned[k] = int(v)
        else:
            cleaned[k] = v
    return json.dumps(cleaned)


def deserialize_features(features_json: str) -> Dict[str, Any]:
    """Deserializes JSON string into a feature dictionary."""
    if not features_json:
        return {}
    return json.loads(features_json)


def generate_synthetic_transactions(n_samples: int = 200, fraud_ratio: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic credit card transaction data strictly for UI demonstration
    and automated testing when raw dataset is missing.
    Matches the exact 30-feature schema of the ULB Credit Card dataset (Time, V1..V28, Amount, Class).
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Generate legitimate data
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    amounts_legit = np.abs(rng.exponential(scale=65.0, size=n_legit)) + 1.0
    times_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    class_legit = np.zeros(n_legit, dtype=int)

    # Generate fraudulent data with distinctive distributional shifts in key PCA components (e.g. V14, V17, V12, V4)
    v_fraud = rng.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
    # Characteristic anomalies in ULB fraud: V14 and V17 typically heavily negative, V4 positive
    v_fraud[:, 13] -= 6.5  # V14
    v_fraud[:, 16] -= 5.0  # V17
    v_fraud[:, 11] -= 5.5  # V12
    v_fraud[:, 3] += 4.5   # V4
    amounts_fraud = np.abs(rng.exponential(scale=180.0, size=n_fraud)) + 10.0
    times_fraud = rng.uniform(0, 172800, size=n_fraud)
    class_fraud = np.ones(n_fraud, dtype=int)

    # Combine
    v_all = np.vstack([v_legit, v_fraud])
    amounts_all = np.concatenate([amounts_legit, amounts_fraud])
    times_all = np.concatenate([times_legit, times_fraud])
    class_all = np.concatenate([class_legit, class_fraud])

    # Shuffle
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([times_all, v_all, amounts_all, class_all])[indices]

    df = pd.DataFrame(data, columns=columns)
    df["Class"] = df["Class"].astype(int)
    return df
