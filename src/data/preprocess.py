"""
Data Preprocessing Pipeline.
Implements chronological train/val/test splitting, robust scaling for Amount and Time,
leaves V1-V28 intact, and provides leak-free transformation artifacts.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from config.settings import MODEL_CONFIG, PATHS
from src.utils.logging_config import logger


class Preprocessor:
    """Preprocesses transaction records for training and real-time inference."""

    def __init__(self, time_col: str = MODEL_CONFIG.TIME_COLUMN, amount_col: str = MODEL_CONFIG.AMOUNT_COLUMN):
        self.time_col = time_col
        self.amount_col = amount_col
        self.pca_cols = MODEL_CONFIG.PCA_FEATURES
        self.all_feature_cols = MODEL_CONFIG.ALL_FEATURES
        self.target_col = MODEL_CONFIG.TARGET_COLUMN
        self.transformer: Optional[ColumnTransformer] = None

    def create_column_transformer(self) -> ColumnTransformer:
        """
        Creates a ColumnTransformer applying RobustScaler to Amount and Time
        (due to extreme financial transaction outliers) while passing through V1-V28.
        """
        # Features to scale: Time, Amount
        scale_features = [self.time_col, self.amount_col]
        # Features to pass through: V1 - V28
        passthrough_features = self.pca_cols

        transformer = ColumnTransformer(
            transformers=[
                ("robust_scaler", RobustScaler(), scale_features),
                ("passthrough", "passthrough", passthrough_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        return transformer

    def split_data_chronological(
        self,
        df: pd.DataFrame,
        val_size: float = MODEL_CONFIG.VAL_SIZE,
        test_size: float = MODEL_CONFIG.TEST_SIZE
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically based on 'Time' to prevent lookahead data leakage.
        Returns: (train_df, val_df, test_df)
        """
        logger.info(f"Splitting dataset chronologically: Train={1.0 - val_size - test_size:.2f}, Val={val_size:.2f}, Test={test_size:.2f}")
        df_sorted = df.sort_values(by=self.time_col).reset_index(drop=True)
        n = len(df_sorted)
        
        train_end = int(n * (1.0 - val_size - test_size))
        val_end = int(n * (1.0 - test_size))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        logger.info(
            f"Chronological split completed: "
            f"Train={len(train_df):,} rows, Val={len(val_df):,} rows, Test={len(test_df):,} rows."
        )
        return train_df, val_df, test_df

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """Fits the scaling transformer on training data only."""
        logger.info("Fitting preprocessing transformer on training data...")
        self.transformer = self.create_column_transformer()
        self.transformer.fit(train_df[self.all_feature_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame to preprocessed feature matrix."""
        if self.transformer is None:
            raise ValueError("Preprocessor has not been fitted. Call fit() or load_pipeline() first.")
        
        # Ensure missing columns are caught or filled
        for col in self.all_feature_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required feature column: {col}")
        
        return self.transformer.transform(df[self.all_feature_cols])

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fits on training data and transforms it."""
        return self.fit(train_df).transform(train_df)

    def save_pipeline(self, artifact_path: Path = PATHS.MODELS_DIR / "preprocessor.joblib") -> Path:
        """Saves fitted transformer artifact."""
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformer, artifact_path)
        logger.info(f"Saved preprocessing pipeline to {artifact_path}")
        return artifact_path

    def load_pipeline(self, artifact_path: Path = PATHS.MODELS_DIR / "preprocessor.joblib") -> "Preprocessor":
        """Loads fitted transformer artifact."""
        if not artifact_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {artifact_path}")
        self.transformer = joblib.load(artifact_path)
        logger.info(f"Loaded preprocessing pipeline from {artifact_path}")
        return self


def get_feature_matrix_and_labels(
    df: pd.DataFrame,
    preprocessor: Preprocessor,
    target_col: str = MODEL_CONFIG.TARGET_COLUMN
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convenience helper to extract preprocessed features and target labels."""
    X = preprocessor.transform(df)
    y = df[target_col].values if target_col in df.columns else None
    return X, y
