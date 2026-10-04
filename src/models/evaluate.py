"""
Model Evaluation and Business Impact Analysis Module.
Computes class-imbalance-appropriate metrics (PR-AUC, Recall, Precision, F1, ROC-AUC),
decision threshold sensitivity curves, and configurable business cost simulations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)
from config.settings import PATHS
from src.utils.logging_config import logger


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Evaluates binary classification performance with particular focus on minority class detection.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # In extreme cases where test set might have only 1 class, handle roc_auc safely
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.50

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Precision-Recall curve points for plotting
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)

    report_dict = {
        "model_name": model_name,
        "decision_threshold": float(threshold),
        "total_test_samples": int(len(y_true)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "metrics": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "false_positive_rate": round(float(fpr), 4)
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        # Downsampled curve coordinates to keep JSON reports lean and snappy
        "pr_curve": {
            "precision": [round(float(p), 4) for p in p_curve[::max(1, len(p_curve) // 100)]],
            "recall": [round(float(r), 4) for r in r_curve[::max(1, len(r_curve) // 100)]]
        },
        "roc_curve": {
            "fpr": [round(float(f), 4) for f in fpr_curve[::max(1, len(fpr_curve) // 100)]],
            "tpr": [round(float(t), 4) for t in tpr_curve[::max(1, len(tpr_curve) // 100)]]
        }
    }
    return report_dict


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Computes precision, recall, F1, and flagged count across various decision thresholds.
    Enables investigation teams to pick optimal operating points based on investigator capacity.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    total = len(y_true)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        flagged = int(tp + fp)

        rows.append({
            "threshold": t,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "flagged_transactions": flagged,
            "flagged_percentage": round((flagged / total) * 100, 2),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })

    return pd.DataFrame(rows)


def simulate_business_costs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_false_positive: float = 15.0,  # e.g. $15 analyst review time / friction
    cost_false_negative: float = 250.0, # e.g. $250 avg net unrecoverable fraud loss
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Simulates financial trade-offs between manual investigation friction (FP)
    and unmitigated fraud loss (FN) at varying decision thresholds.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        fp_cost = fp * cost_false_positive
        fn_cost = fn * cost_false_negative
        total_cost = fp_cost + fn_cost

        rows.append({
            "threshold": t,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "investigation_friction_cost": round(fp_cost, 2),
            "fraud_loss_cost": round(fn_cost, 2),
            "total_estimated_cost": round(total_cost, 2)
        })

    return pd.DataFrame(rows)


def save_evaluation_report(
    eval_dict: Dict[str, Any],
    output_path: Path = PATHS.REPORTS_DIR / "model_evaluation_report.json"
) -> Path:
    """Saves evaluation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)
    logger.info(f"Model evaluation report saved to {output_path}")
    return output_path
