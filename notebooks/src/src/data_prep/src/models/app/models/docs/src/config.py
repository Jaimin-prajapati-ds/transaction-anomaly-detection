"""Configuration file for anomaly detection system."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model parameters
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.1,
    "random_state": 42
}

PCA_PARAMS = {
    "n_components": 5,
    "random_state": 42
}

# Feature columns
FEATURE_COLS = [
    "amount",
    "hour",
    "distance_from_home",
    "is_international",
    "transaction_speed"
]

# Thresholds
ANOMALY_THRESHOLD = 0.5
