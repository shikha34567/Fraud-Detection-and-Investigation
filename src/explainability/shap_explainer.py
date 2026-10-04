"""
Explainable AI (XAI) Module using SHAP (SHapley Additive exPlanations).
Computes global feature importance and localized transaction attributions.
Distinguishes between factors elevating vs reducing fraud risk with human-readable narratives.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

from config.settings import PATHS, MODEL_CONFIG
from src.utils.logging_config import logger
from src.data.preprocess import Preprocessor


class ShapExplainer:
    """
    Computes explainability insights using TreeExplainer or standard Explainer.
    
    Ethical / Causal Disclaimer:
    SHAP values quantify how strongly each feature nudges the mathematical model's
    prediction toward or away from a fraud classification. They reflect correlation
    within the training distribution, NOT forensic proof of malicious intent.
    """

    def __init__(
        self,
        model_path: Path = PATHS.MODELS_DIR / "best_model.joblib",
        preprocessor_path: Path = PATHS.MODELS_DIR / "preprocessor.joblib"
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.explainer: Optional[shap.Explainer] = None
        self.feature_names = ["Time", "Amount"] + MODEL_CONFIG.PCA_FEATURES

    def initialize_explainer(self, background_samples: Optional[np.ndarray] = None) -> bool:
        """Initializes appropriate SHAP explainer."""
        if not self.model_path.exists() or not self.preprocessor_path.exists():
            logger.warning("Models not found; cannot initialize SHAP explainer.")
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = Preprocessor().load_pipeline(self.preprocessor_path)

            # Check if tree-based
            model_type = type(self.model).__name__
            logger.info(f"Initializing SHAP explainer for model type: {model_type}")

            if "XGB" in model_type or "Forest" in model_type or "Tree" in model_type:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Linear or kernel explainer with background sample
                if background_samples is None:
                    # Synthetic background baseline
                    background_samples = np.zeros((20, len(MODEL_CONFIG.ALL_FEATURES)))
                self.explainer = shap.LinearExplainer(self.model, background_samples)

            logger.info("SHAP explainer initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}", exc_info=True)
            return False

    def explain_transaction(
        self,
        transaction_features: Dict[str, Any] | pd.Series,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generates individual SHAP explanation for a single transaction.
        Separates top risk-increasing and top risk-decreasing drivers.
        """
        if self.explainer is None:
            if not self.initialize_explainer():
                raise RuntimeError("SHAP explainer could not be initialized.")

        # Convert to DataFrame and preprocess
        if isinstance(transaction_features, dict):
            df_row = pd.DataFrame([transaction_features])
        else:
            df_row = pd.DataFrame([transaction_features.to_dict()])

        X = self.preprocessor.transform(df_row)

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(X)

        # Handle various output shapes from different models/shap versions
        if isinstance(raw_shap, list):
            # Binary classification list: [class_0_values, class_1_values]
            shap_vec = raw_shap[1][0]
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:  # (samples, features, classes)
                shap_vec = raw_shap[0, :, 1]
            elif raw_shap.ndim == 2:  # (samples, features)
                shap_vec = raw_shap[0]
            else:
                shap_vec = raw_shap
        else:
            shap_vec = np.array(raw_shap)

        # Feature names in transformer order: RobustScaler on (Time, Amount), then passthrough (V1-V28)
        feature_labels = ["Time", "Amount"] + MODEL_CONFIG.PCA_FEATURES

        feature_contributions = []
        for name, val, raw_val in zip(feature_labels, shap_vec, X[0]):
            feature_contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4),
                "transformed_value": round(float(raw_val), 4),
                "impact": "Increases Fraud Risk" if val > 0 else "Decreases Fraud Risk"
            })

        # Sort by absolute SHAP value
        sorted_by_abs = sorted(feature_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_risk_increasing = [f for f in sorted_by_abs if f["shap_value"] > 0][:top_n]
        top_risk_decreasing = [f for f in sorted_by_abs if f["shap_value"] < 0][:top_n]

        # Generate human-readable explanation narrative
        narrative_parts = []
        if top_risk_increasing:
            inc_names = [f"{item['feature']} (+{item['shap_value']:.2f})" for item in top_risk_increasing[:3]]
            narrative_parts.append(f"Primary risk drivers elevating fraud score: {', '.join(inc_names)}.")
        if top_risk_decreasing:
            dec_names = [f"{item['feature']} ({item['shap_value']:.2f})" for item in top_risk_decreasing[:2]]
            narrative_parts.append(f"Mitigating factors: {', '.join(dec_names)}.")

        narrative = " ".join(narrative_parts) if narrative_parts else "No strong directional deviations detected."

        return {
            "all_contributions": sorted_by_abs,
            "top_risk_increasing": top_risk_increasing,
            "top_risk_decreasing": top_risk_decreasing,
            "narrative_explanation": narrative
        }

    def compute_global_importance(self, df_sample: pd.DataFrame, max_rows: int = 150) -> pd.DataFrame:
        """Computes mean absolute SHAP values across a batch of sample transactions."""
        if self.explainer is None:
            if not self.initialize_explainer():
                raise RuntimeError("SHAP explainer could not be initialized.")

        sample = df_sample.head(max_rows)
        X = self.preprocessor.transform(sample)
        raw_shap = self.explainer.shap_values(X)

        if isinstance(raw_shap, list):
            shap_matrix = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_matrix = raw_shap[:, :, 1]
        else:
            shap_matrix = raw_shap

        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        feature_labels = ["Time", "Amount"] + MODEL_CONFIG.PCA_FEATURES

        df_importance = pd.DataFrame({
            "feature": feature_labels,
            "mean_abs_shap": np.round(mean_abs, 4)
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return df_importance


shap_explainer = ShapExplainer()
