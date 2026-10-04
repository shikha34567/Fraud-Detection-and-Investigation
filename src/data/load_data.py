"""
Dataset loading module for ULB Credit Card Fraud Detection dataset.
Handles missing files gracefully, provides demo data generation, and provides status flags.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from config.settings import PATHS
from src.utils.logging_config import logger
from src.utils.helpers import generate_synthetic_transactions


class DataLoader:
    """Manages loading of the ULB Credit Card Fraud dataset or fallback demo data."""

    def __init__(self, raw_path: Path = PATHS.RAW_DATA_FILE):
        self.raw_path = raw_path

    def is_dataset_available(self) -> bool:
        """Checks if the actual raw CSV file exists and has non-zero size."""
        return self.raw_path.exists() and self.raw_path.stat().st_size > 1024

    def load_raw_data(self, use_demo_if_missing: bool = False, demo_samples: int = 2500) -> Tuple[pd.DataFrame, bool]:
        """
        Loads the raw CSV dataset.
        If file is missing and use_demo_if_missing is True, returns synthetic demo data.
        Returns:
            Tuple[pd.DataFrame, bool]: (DataFrame, is_demo_flag)
        """
        if self.is_dataset_available():
            logger.info(f"Loading raw dataset from {self.raw_path}")
            df = pd.read_csv(self.raw_path)
            logger.info(f"Successfully loaded {len(df):,} transactions from raw dataset.")
            return df, False

        if use_demo_if_missing:
            logger.warning(
                f"Raw dataset not found at {self.raw_path}. "
                f"Generating {demo_samples} synthetic transactions strictly for UI/Testing demonstration."
            )
            df = generate_synthetic_transactions(n_samples=demo_samples, fraud_ratio=0.03)
            return df, True

        raise FileNotFoundError(
            f"Dataset not found at expected path: {self.raw_path}\n"
            "Please download 'creditcard.csv' from Kaggle:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place it inside '{self.raw_path.parent.resolve()}'."
        )


def load_dataset(use_demo_if_missing: bool = True) -> Tuple[pd.DataFrame, bool]:
    """Helper function to load dataset with fallback."""
    loader = DataLoader()
    return loader.load_raw_data(use_demo_if_missing=use_demo_if_missing)
