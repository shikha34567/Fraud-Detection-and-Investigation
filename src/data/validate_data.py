"""
Data validation module.
Inspects schemas, column presence, null values, duplicates, and class imbalance.
Generates comprehensive quality audit reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from config.settings import MODEL_CONFIG, PATHS
from src.utils.logging_config import logger


class DataValidator:
    """Validates structure and statistical integrity of transaction datasets."""

    def __init__(self, target_col: str = MODEL_CONFIG.TARGET_COLUMN):
        self.target_col = target_col
        self.expected_columns = MODEL_CONFIG.ALL_FEATURES + [self.target_col]

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates column presence and types."""
        missing_cols = [c for c in self.expected_columns if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in self.expected_columns]
        
        is_valid = len(missing_cols) == 0
        return {
            "is_valid": is_valid,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "total_columns": len(df.columns),
            "expected_columns_count": len(self.expected_columns),
        }

    def generate_quality_report(self, df: pd.DataFrame, is_demo: bool = False) -> Dict[str, Any]:
        """
        Performs in-depth data quality audit including missing values,
        duplicates, class distribution, and numerical health.
        """
        logger.info("Executing comprehensive data quality audit...")
        schema_info = self.validate_schema(df)

        # Missing values
        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())
        missing_by_col = {k: int(v) for k, v in missing_counts[missing_counts > 0].items()}

        # Duplicates
        duplicate_count = int(df.duplicated().sum())

        # Target distribution
        has_target = self.target_col in df.columns
        target_stats = {}
        if has_target:
            val_counts = df[self.target_col].value_counts().to_dict()
            total_records = len(df)
            fraud_count = int(val_counts.get(1, 0))
            legit_count = int(val_counts.get(0, 0))
            fraud_percentage = round((fraud_count / total_records) * 100, 4) if total_records else 0.0
            imbalance_ratio = round(legit_count / fraud_count, 1) if fraud_count > 0 else 0

            target_stats = {
                "legitimate_count": legit_count,
                "fraud_count": fraud_count,
                "fraud_percentage": fraud_percentage,
                "imbalance_ratio": f"{imbalance_ratio}:1" if fraud_count > 0 else "N/A"
            }

        # Amount statistics
        amount_stats = {}
        if MODEL_CONFIG.AMOUNT_COLUMN in df.columns:
            amt = df[MODEL_CONFIG.AMOUNT_COLUMN]
            amount_stats = {
                "min": round(float(amt.min()), 2),
                "max": round(float(amt.max()), 2),
                "mean": round(float(amt.mean()), 2),
                "median": round(float(amt.median()), 2),
                "std": round(float(amt.std()), 2),
                "zero_amount_count": int((amt == 0).sum()),
                "negative_amount_count": int((amt < 0).sum())
            }

        # Time statistics
        time_stats = {}
        if MODEL_CONFIG.TIME_COLUMN in df.columns:
            t = df[MODEL_CONFIG.TIME_COLUMN]
            time_stats = {
                "min_seconds": float(t.min()),
                "max_seconds": float(t.max()),
                "duration_hours": round(float((t.max() - t.min()) / 3600), 2)
            }

        # Warnings collection
        warnings: List[str] = []
        if is_demo:
            warnings.append("Operating in SYNTHETIC DEMO MODE: Dataset is generated for interface testing.")
        if total_missing > 0:
            warnings.append(f"Detected {total_missing} missing values across features.")
        if duplicate_count > 0:
            warnings.append(f"Detected {duplicate_count} duplicate rows in dataset.")
        if amount_stats.get("negative_amount_count", 0) > 0:
            warnings.append("Detected transactions with negative monetary amounts.")
        if target_stats.get("fraud_percentage", 1) < 0.5:
            warnings.append(f"Severe class imbalance: Fraud accounts for only {target_stats.get('fraud_percentage')}% of data.")

        report = {
            "is_demo_data": is_demo,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "schema_validation": schema_info,
            "missing_values": {
                "total_missing": total_missing,
                "columns_with_missing": missing_by_col
            },
            "duplicate_records": duplicate_count,
            "target_distribution": target_stats,
            "amount_statistics": amount_stats,
            "time_statistics": time_stats,
            "validation_warnings": warnings,
            "status": "PASSED" if schema_info["is_valid"] and total_missing == 0 else "WARNING"
        }

        return report

    def save_report(self, report: Dict[str, Any], output_path: Path = PATHS.REPORTS_DIR / "data_quality_report.json") -> Path:
        """Saves report to json file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Data quality report saved to {output_path}")
        return output_path
