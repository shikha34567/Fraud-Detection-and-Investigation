"""
Transaction Simulation Engine.
Streams simulated financial transactions, scores them in real time using the ML inference pipeline,
and persists records into the SQL database with audit labels.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from config.settings import MODEL_CONFIG
from src.utils.helpers import generate_transaction_reference, generate_synthetic_transactions
from src.data.load_data import load_dataset
from src.models.predict import predictor
from src.database.connection import get_db_session
from src.database.crud import insert_transaction_with_prediction
from src.utils.logging_config import logger


class TransactionSimulator:
    """
    Simulates streaming financial transactions from historical data or synthetic generator.
    
    Disclaimer:
    Transactions generated or streamed are for demonstration, simulation, and model stress-testing.
    They do NOT represent a live socket to credit card payment networks.
    """

    def __init__(self):
        self._source_df: Optional[pd.DataFrame] = None
        self._current_index: int = 0
        self.is_demo_source: bool = False

    def load_stream_source(self) -> None:
        """Loads data source for streaming."""
        df, is_demo = load_dataset(use_demo_if_missing=True)
        # Drop target if present to simulate raw unlabelled feed
        self._source_df = df.copy()
        self.is_demo_source = is_demo
        self._current_index = 0
        logger.info(f"Loaded simulation stream pool: {len(self._source_df):,} records (demo={is_demo})")

    def get_next_batch(self, batch_size: int = 5, inject_fraud_probability: float = 0.20) -> List[Dict[str, Any]]:
        """
        Retrieves next batch of transactions.
        If at the end of stream, loops or generates fresh records.
        """
        if self._source_df is None or len(self._source_df) == 0:
            self.load_stream_source()

        batch_records = []
        n_available = len(self._source_df)

        for _ in range(batch_size):
            if self._current_index >= n_available:
                self._current_index = 0  # Loop back or reshuffle

            row = self._source_df.iloc[self._current_index].to_dict()
            self._current_index += 1

            # Strip target column from input features to simulate realistic incoming payload
            ground_truth = row.pop(MODEL_CONFIG.TARGET_COLUMN, None)
            features = {k: v for k, v in row.items()}
            txn_ref = generate_transaction_reference()

            batch_records.append({
                "transaction_reference": txn_ref,
                "time": float(features.get("Time", 0.0)),
                "amount": float(features.get("Amount", 0.0)),
                "features": features,
                "ground_truth": ground_truth,
                "is_simulated": True
            })

        return batch_records

    def process_and_persist_batch(self, batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Pulls a batch, scores each transaction using FraudPredictor,
        and saves into database.
        """
        batch = self.get_next_batch(batch_size=batch_size)
        processed_results = []

        # Convert to DataFrame for batch inference
        feature_rows = [b["features"] for b in batch]
        df_batch = pd.DataFrame(feature_rows)

        scored_df = predictor.predict_dataframe(df_batch)

        with get_db_session() as session:
            for idx, item in enumerate(batch):
                pred_row = scored_df.iloc[idx]
                fraud_prob = float(pred_row["fraud_probability"])
                risk_score = float(pred_row["risk_score"])
                risk_cat = str(pred_row["risk_category"])
                predicted_label = int(pred_row["predicted_fraud"])
                anomaly_score = float(pred_row.get("anomaly_score", 0.0))

                insert_transaction_with_prediction(
                    db=session,
                    transaction_ref=item["transaction_reference"],
                    transaction_time=item["time"],
                    amount=item["amount"],
                    features=item["features"],
                    fraud_probability=fraud_prob,
                    risk_score=risk_score,
                    risk_category=risk_cat,
                    predicted_label=predicted_label,
                    anomaly_score=anomaly_score,
                    model_version="v1.0.0-xgb"
                )

                processed_results.append({
                    "reference": item["transaction_reference"],
                    "time": item["time"],
                    "amount": item["amount"],
                    "fraud_probability": fraud_prob,
                    "risk_score": risk_score,
                    "risk_category": risk_cat,
                    "predicted_label": predicted_label,
                    "anomaly_score": anomaly_score,
                    "is_simulated": True
                })

        return processed_results


simulator = TransactionSimulator()
