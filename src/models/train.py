"""
Model Training and Comparison Pipeline.
Trains Baseline Logistic Regression, Random Forest, XGBoost, and Isolation Forest.
Handles class imbalance strictly within training sets, compares performance,
and persists artifacts in artifacts/models/.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from config.settings import MODEL_CONFIG, PATHS
from src.utils.logging_config import logger
from src.data.load_data import load_dataset
from src.data.preprocess import Preprocessor
from src.models.anomaly_detection import AnomalyDetector
from src.models.evaluate import evaluate_binary_classifier, save_evaluation_report


def train_models(
    use_demo_if_missing: bool = True,
    sample_limit: int | None = None
) -> Dict[str, Any]:
    """
    Executes the complete training workflow:
    1. Loads dataset (or demo fallback if absent)
    2. Chronologically splits into Train, Validation, Test sets
    3. Fits Preprocessor exclusively on Train set
    4. Calculates class weights for minority fraud cases
    5. Trains Logistic Regression, Random Forest, XGBoost, Isolation Forest
    6. Evaluates all models on held-out Test set
    7. Identifies and persists the best performing model
    """
    logger.info("Starting Machine Learning Training Pipeline...")

    # Step 1: Load Data
    raw_df, is_demo = load_dataset(use_demo_if_missing=use_demo_if_missing)
    if sample_limit and len(raw_df) > sample_limit:
        raw_df = raw_df.iloc[:sample_limit].copy()

    # Step 2: Chronological Split
    preprocessor = Preprocessor()
    train_df, val_df, test_df = preprocessor.split_data_chronological(raw_df)

    # Step 3: Fit Preprocessor on Train set only
    X_train = preprocessor.fit_transform(train_df)
    preprocessor.save_pipeline()

    # Transform Validation and Test sets without leakage
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df[MODEL_CONFIG.TARGET_COLUMN].values
    y_val = val_df[MODEL_CONFIG.TARGET_COLUMN].values
    y_test = test_df[MODEL_CONFIG.TARGET_COLUMN].values

    # Step 4: Class Imbalance Ratio
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(1, n_pos))
    logger.info(f"Class imbalance on training set: {n_neg} Legit vs {n_pos} Fraud (scale_pos_weight={scale_pos_weight:.2f})")

    # Step 5: Model Definitions
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=MODEL_CONFIG.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=MODEL_CONFIG.RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=MODEL_CONFIG.RANDOM_STATE,
            n_jobs=-1
        )
    }

    comparison_results = {}
    fitted_models = {}

    for name, model in models.items():
        logger.info(f"--> Training {name}...")
        if name == "XGBoost":
            # Pass validation set for early stopping if desired
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        fitted_models[name] = model

        # Predict probabilities on held-out test set
        y_prob = model.predict_proba(X_test)[:, 1]
        eval_result = evaluate_binary_classifier(y_test, y_prob, threshold=0.50, model_name=name)
        comparison_results[name] = eval_result
        logger.info(
            f"[{name}] Test PR-AUC: {eval_result['metrics']['pr_auc']:.4f} | "
            f"Recall: {eval_result['metrics']['recall']:.4f} | "
            f"Precision: {eval_result['metrics']['precision']:.4f} | "
            f"F1: {eval_result['metrics']['f1_score']:.4f}"
        )

    # Step 6: Train Unsupervised Isolation Forest Anomaly Detector
    logger.info("--> Training Isolation Forest for Unsupervised Outlier Screening...")
    # Train anomaly detector on standard training distribution
    iso_forest = AnomalyDetector(contamination=0.015, random_state=MODEL_CONFIG.RANDOM_STATE)
    iso_forest.fit(X_train)
    iso_forest.save()

    # Step 7: Select Best Supervised Model (Ranked by PR-AUC, F1, and model preference: XGBoost > Random Forest > Logistic Regression)
    preference_order = {"XGBoost": 3, "Random Forest": 2, "Logistic Regression": 1}
    best_model_name = max(
        comparison_results.keys(),
        key=lambda k: (
            comparison_results[k]["metrics"]["pr_auc"],
            comparison_results[k]["metrics"]["f1_score"],
            preference_order.get(k, 0)
        )
    )
    best_model = fitted_models[best_model_name]
    logger.info(f"Best selected supervised model: '{best_model_name}'")

    # Persist Best Model Artifact
    best_model_path = PATHS.MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    logger.info(f"Persisted best model to {best_model_path}")

    # Persist Model Metadata
    metadata = {
        "best_model_name": best_model_name,
        "is_demo_data": is_demo,
        "total_train_samples": len(train_df),
        "total_test_samples": len(test_df),
        "class_imbalance_ratio": f"{n_neg}:{n_pos}",
        "feature_count": len(MODEL_CONFIG.ALL_FEATURES),
        "feature_names": MODEL_CONFIG.ALL_FEATURES,
        "metrics_summary": {k: v["metrics"] for k, v in comparison_results.items()}
    }

    metadata_path = PATHS.MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save comprehensive evaluation report
    save_evaluation_report({
        "comparison": comparison_results,
        "best_model": best_model_name,
        "is_demo": is_demo,
        "sample_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        }
    })

    logger.info("Machine Learning Pipeline execution finished successfully!")
    return {
        "comparison": comparison_results,
        "best_model": best_model_name,
        "is_demo": is_demo
    }


if __name__ == "__main__":
    train_models(use_demo_if_missing=True)
