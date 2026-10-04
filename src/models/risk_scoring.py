"""
Risk Scoring Module.
Converts model-predicted fraud probabilities into an interpretable 0-100 risk score,
assigns transparent risk categories based on configurable thresholds,
and documents the mathematical derivation.
"""

from typing import Dict, Any, Union
import numpy as np
from config.settings import RISK_THRESHOLDS


class RiskScorer:
    """
    Transparent risk scoring engine.
    
    Derivation:
    The Risk Score (0 - 100) is a non-linear mapping of the model's calibrated
    posterior fraud probability P(Fraud | X):
        Risk Score = round(P(Fraud | X) * 100, 2)
    
    Categories:
    - 0.0 <= Score < 25.0:  Low Risk (Standard approval, normal processing)
    - 25.0 <= Score < 60.0: Medium Risk (Elevated scrutiny, additional verification recommended)
    - 60.0 <= Score < 85.0: High Risk (Flagged for urgent investigator review)
    - 85.0 <= Score <= 100.0: Critical Risk (Immediate hold / block recommended)
    """

    def __init__(
        self,
        low_upper: float = RISK_THRESHOLDS.LOW_UPPER,
        medium_upper: float = RISK_THRESHOLDS.MEDIUM_UPPER,
        high_upper: float = RISK_THRESHOLDS.HIGH_UPPER
    ):
        self.low_upper = low_upper
        self.medium_upper = medium_upper
        self.high_upper = high_upper

    def compute_risk_score(self, fraud_probability: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculates 0-100 risk score from probability."""
        if isinstance(fraud_probability, np.ndarray):
            clipped = np.clip(fraud_probability, 0.0, 1.0)
            return np.round(clipped * 100.0, 2)
        
        prob = max(0.0, min(1.0, float(fraud_probability)))
        return round(prob * 100.0, 2)

    def assign_risk_category(self, risk_score: float) -> str:
        """Assigns qualitative risk category from risk score."""
        score = float(risk_score)
        if score < self.low_upper:
            return "Low Risk"
        elif score < self.medium_upper:
            return "Medium Risk"
        elif score < self.high_upper:
            return "High Risk"
        else:
            return "Critical Risk"

    def score_transaction(
        self,
        fraud_probability: float,
        anomaly_score: float | None = None
    ) -> Dict[str, Any]:
        """
        Generates complete risk scoring payload for a single transaction.
        Note: The anomaly score is reported alongside but NOT combined into the
        calibrated supervised probability to preserve mathematical validity.
        """
        score = float(self.compute_risk_score(fraud_probability))
        category = self.assign_risk_category(score)
        is_flagged = score >= self.low_upper  # Flagged if above Low Risk

        result = {
            "fraud_probability": round(float(fraud_probability), 4),
            "risk_score": score,
            "risk_category": category,
            "is_flagged": is_flagged,
            "anomaly_score": round(float(anomaly_score), 4) if anomaly_score is not None else None,
            "scoring_methodology": "Calibrated Posterior Probability Scaled [0-100]"
        }
        return result


risk_scorer = RiskScorer()
