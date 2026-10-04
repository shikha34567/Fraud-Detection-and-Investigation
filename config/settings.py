"""
Application and ML Pipeline Configuration Settings.
Loads environment variables or defaults for reproducible deployment.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class RiskThresholds:
    """Configurable risk thresholds (0 to 100)."""
    LOW_UPPER: float = 25.0       # 0.0 - 25.0: Low Risk
    MEDIUM_UPPER: float = 60.0    # 25.0 - 60.0: Medium Risk
    HIGH_UPPER: float = 85.0      # 60.0 - 85.0: High Risk
    # >= 85.0: Critical Risk

    DECISION_THRESHOLD: float = 0.50  # Probability threshold for binary fraud flag


@dataclass
class PathSettings:
    """Project directory paths."""
    BASE: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    REPORTS_DIR: Path = BASE_DIR / "data" / "reports"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    MODELS_DIR: Path = BASE_DIR / "artifacts" / "models"
    EXPLAINERS_DIR: Path = BASE_DIR / "artifacts" / "explainers"
    DATABASE_PATH: Path = BASE_DIR / "data" / "fraud_detection.db"
    RAW_DATA_FILE: Path = BASE_DIR / "data" / "raw" / "creditcard.csv"


@dataclass
class ModelSettings:
    """Machine learning hyperparameters and feature specifications."""
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.10

    # Dataset Feature Schema
    TARGET_COLUMN: str = "Class"
    TIME_COLUMN: str = "Time"
    AMOUNT_COLUMN: str = "Amount"
    PCA_FEATURES: List[str] = field(
        default_factory=lambda: [f"V{i}" for i in range(1, 29)]
    )

    @property
    def ALL_FEATURES(self) -> List[str]:
        return [self.TIME_COLUMN] + self.PCA_FEATURES + [self.AMOUNT_COLUMN]

    # Models list
    MODELS: List[str] = field(
        default_factory=lambda: [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Isolation Forest"
        ]
    )


@dataclass
class DatabaseSettings:
    """SQLAlchemy Database connection settings."""
    DB_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{PathSettings.DATABASE_PATH.as_posix()}"
    )


# Instantiate singletons for imports
PATHS = PathSettings()
RISK_THRESHOLDS = RiskThresholds()
MODEL_CONFIG = ModelSettings()
DB_CONFIG = DatabaseSettings()

# Ensure directories exist
for directory in [
    PATHS.RAW_DATA_DIR,
    PATHS.PROCESSED_DATA_DIR,
    PATHS.REPORTS_DIR,
    PATHS.MODELS_DIR,
    PATHS.EXPLAINERS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
