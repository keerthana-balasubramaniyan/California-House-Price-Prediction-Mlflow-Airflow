"""Central configuration for the California House Price Prediction project.

All paths are derived from this file's location, so the project is portable
and contains no machine-specific absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for _directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# Train/test split fraction used for the held-out test set.
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
# Airflow tasks and the MLflow UI must agree on this URI, otherwise runs end up
# in different stores.  It can be overridden with the MLFLOW_TRACKING_URI
# environment variable (see .env.example).
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

MLFLOW_EXPERIMENT_NAME = "california-house-price-prediction"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
TARGET_COLUMN = "MedHouseVal"
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# --------------------------------------------------------------------------
# Model hyper-parameters (deliberately modest so the pipeline runs fast locally)
# --------------------------------------------------------------------------
MODEL_PARAMS: dict[str, dict[str, object]] = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": RANDOM_STATE,
    },
}
