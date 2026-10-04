"""
Anomaly Detection Module using Isolation Forest.
Detects statistical outliers in unlabelled or legitimate transaction streams.
Produces distinct anomaly scores kept clearly separate from supervised probabilities.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from config.settings import PATHS
from src.utils.logging_config import logger


class AnomalyDetector:
    """
    Unsupervised Anomaly Detector using Isolation Forest.
    
    Important Distinction:
    Anomaly detection finds data points structurally distant from standard transaction distributions.
    An anomalous transaction is not guaranteed to be fraudulent (e.g. holiday gift spending, luxury travel),
    and a fraudulent transaction might intentionally blend into standard patterns.
    Therefore, the anomaly score is strictly presented as an independent investigation signal.
    """

    def __init__(self, contamination: float = 0.01, random_state: int = 42, n_estimators: int = 100):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fits the Isolation Forest on feature matrix."""
        logger.info(f"Training Isolation Forest with contamination={self.contamination} on {len(X)} samples...")
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Isolation Forest training complete.")
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores.
        Note: IsolationForest.decision_function yields negative scores for outliers and positive for inliers.
        We invert it so higher score = higher degree of abnormality.
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector has not been fitted.")
        raw_scores = self.model.decision_function(X)
        # Invert so higher score represents higher anomaly level
        anomaly_scores = -raw_scores
        return anomaly_scores

    def predict_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns binary anomaly labels (-1 anomaly, 1 normal) and numerical anomaly scores.
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector has not been fitted.")
        preds = self.model.predict(X)  # -1 for outliers, 1 for inliers
        scores = self.score_samples(X)
        is_anomaly = (preds == -1)
        return is_anomaly, scores

    def save(self, artifact_path: Path = PATHS.MODELS_DIR / "isolation_forest.joblib") -> Path:
        """Persists trained model artifact."""
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, artifact_path)
        logger.info(f"Saved Isolation Forest artifact to {artifact_path}")
        return artifact_path

    def load(self, artifact_path: Path = PATHS.MODELS_DIR / "isolation_forest.joblib") -> "AnomalyDetector":
        """Loads model artifact."""
        if not artifact_path.exists():
            raise FileNotFoundError(f"Anomaly detector artifact not found at {artifact_path}")
        self.model = joblib.load(artifact_path)
        self.is_fitted = True
        logger.info(f"Loaded Isolation Forest artifact from {artifact_path}")
        return self
