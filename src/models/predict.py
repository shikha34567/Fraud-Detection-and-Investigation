"""
Inference and Prediction Pipeline.
Loads preprocessor, best classifier, and anomaly detector.
Produces fraud probabilities, calibrated risk scores, and anomaly alerts.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import joblib
import numpy as np
import pandas as pd
from config.settings import PATHS, MODEL_CONFIG
from src.utils.logging_config import logger
from src.data.preprocess import Preprocessor
from src.models.risk_scoring import risk_scorer, RiskScorer


class FraudPredictor:
    """End-to-end inference service for real-time transactions."""

    def __init__(
        self,
        model_path: Path = PATHS.MODELS_DIR / "best_model.joblib",
        preprocessor_path: Path = PATHS.MODELS_DIR / "preprocessor.joblib",
        anomaly_path: Path = PATHS.MODELS_DIR / "isolation_forest.joblib",
        scorer: RiskScorer = risk_scorer
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.anomaly_path = anomaly_path
        self.scorer = scorer

        self.model = None
        self.preprocessor = None
        self.anomaly_detector = None
        self.is_loaded = False

    def check_artifacts_available(self) -> bool:
        """Verifies whether trained model files exist."""
        return self.model_path.exists() and self.preprocessor_path.exists()

    def load_artifacts(self) -> bool:
        """Loads model and preprocessing artifacts into memory."""
        if not self.check_artifacts_available():
            logger.warning(
                f"Model artifacts missing! Expected '{self.model_path}' and '{self.preprocessor_path}'. "
                "Please run 'python -m src.models.train' first."
            )
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = Preprocessor().load_pipeline(self.preprocessor_path)

            if self.anomaly_path.exists():
                self.anomaly_detector = joblib.load(self.anomaly_path)

            self.is_loaded = True
            logger.info("FraudPredictor successfully loaded model and preprocessor artifacts.")
            return True
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)
            return False

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of raw transactions, outputs enriched DataFrame
        with fraud_probability, risk_score, risk_category, and anomaly_score.
        """
        if not self.is_loaded:
            if not self.load_artifacts():
                raise RuntimeError(
                    "Model artifacts are not loaded. Run 'python -m src.models.train' to build them."
                )

        # Preprocess features
        X = self.preprocessor.transform(df)

        # Supervised probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.predict(X)

        # Risk scores and categories
        risk_scores = self.scorer.compute_risk_score(probs)
        risk_categories = [self.scorer.assign_risk_category(s) for s in risk_scores]

        # Anomaly scores
        anomaly_scores = None
        if self.anomaly_detector is not None:
            # -decision_function so higher is more anomalous
            raw_dec = self.anomaly_detector.decision_function(X)
            anomaly_scores = np.round(-raw_dec, 4)

        result_df = df.copy()
        result_df["fraud_probability"] = np.round(probs, 4)
        result_df["risk_score"] = risk_scores
        result_df["risk_category"] = risk_categories
        result_df["predicted_fraud"] = (probs >= 0.50).astype(int)

        if anomaly_scores is not None:
            result_df["anomaly_score"] = anomaly_scores
        else:
            result_df["anomaly_score"] = 0.0

        return result_df

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single raw transaction dictionary."""
        df = pd.DataFrame([features])
        res_df = self.predict_dataframe(df)
        row = res_df.iloc[0].to_dict()
        return {
            "fraud_probability": float(row["fraud_probability"]),
            "risk_score": float(row["risk_score"]),
            "risk_category": str(row["risk_category"]),
            "predicted_fraud": int(row["predicted_fraud"]),
            "anomaly_score": float(row["anomaly_score"])
        }


predictor = FraudPredictor()
