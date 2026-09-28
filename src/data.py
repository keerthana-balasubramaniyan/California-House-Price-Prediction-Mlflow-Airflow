"""Dataset loading, validation and train/test splitting.

This module contains all the data-related logic so the Airflow DAG only has to
orchestrate it.  Nothing here knows about Airflow or MLflow.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)

LOGGER = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the dataset does not satisfy the expected structure."""


def load_dataset(data_home: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset as a pandas feature/target pair.

    The dataset ships with scikit-learn, so no manual download is required.
    Results are cached under ``data/raw`` so repeated Airflow task runs are fast
    and work offline.
    """
    cache_path = RAW_DATA_DIR / "california_housing.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        LOGGER.info("Loading cached dataset from %s", cache_path)
        with cache_path.open("rb") as handle:
            features, target = pickle.load(handle)
    else:
        LOGGER.info("Fetching California Housing dataset via sklearn...")
        try:
            bunch = fetch_california_housing(data_home=data_home)
        except Exception as exc:  # pragma: no cover - network/filesystem failure
            raise RuntimeError(
                f"Failed to load the California Housing dataset: {exc}"
            ) from exc
        features = pd.DataFrame(
            bunch.data, columns=list(bunch.feature_names)
        )
        target = pd.Series(bunch.target, name=TARGET_COLUMN, dtype="float64")
        with cache_path.open("wb") as handle:
            pickle.dump((features, target), handle)
        LOGGER.info("Dataset cached to %s", cache_path)

    LOGGER.info(
        "Loaded dataset: %d samples, %d features", len(features), features.shape[1]
    )
    return features, target


def validate_dataset(
    features: pd.DataFrame, target: pd.Series, expected_features: list[str] | None = None
) -> dict[str, object]:
    """Validate the dataset structure and return a short summary.

    Raises ``DataValidationError`` with an explicit message on any failure.
    """
    expected_features = expected_features or FEATURE_COLUMNS
    report: dict[str, object] = {}

    if features is None or target is None:
        raise DataValidationError("Features and target must not be None.")

    if features.empty:
        raise DataValidationError("Feature dataframe is empty.")
    if target.empty:
        raise DataValidationError("Target series is empty.")

    if len(features) != len(target):
        raise DataValidationError(
            f"Row count mismatch: {len(features)} feature rows vs "
            f"{len(target)} target rows."
        )

    missing = [column for column in expected_features if column not in features.columns]
    if missing:
        raise DataValidationError(
            f"Missing expected feature column(s): {missing}. "
            f"Present columns: {list(features.columns)}"
        )

    if not np.issubdtype(features.to_numpy().dtype, np.number):
        raise DataValidationError(
            f"Non-numeric data found in features (dtype={features.dtypes.unique()})."
        )

    if not np.issubdtype(target.to_numpy().dtype, np.number):
        raise DataValidationError(
            f"Non-numeric data found in target (dtype={target.dtype})."
        )

    missing_values = int(features.isna().sum().sum()) + int(target.isna().sum())
    if missing_values:
        raise DataValidationError(
            f"Dataset contains {missing_values} missing value(s); "
            "this dataset is expected to be complete."
        )

    if features.empty or len(features) < 100:
        raise DataValidationError(
            f"Dataset has only {len(features)} rows, too few to train on."
        )

    report = {
        "n_samples": int(len(features)),
        "n_features": int(features.shape[1]),
        "feature_names": list(features.columns),
        "target_name": TARGET_COLUMN,
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
        "n_missing_values": missing_values,
        "status": "passed",
    }
    LOGGER.info("Dataset validation passed: %s", report)
    return report


def prepare_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test partitions.

    The split happens before any fitting, so test data can never influence
    model training.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    LOGGER.info(
        "Train/Test split (test_size=%.2f, random_state=%d): "
        "X_train=%s y_train=%s X_test=%s y_test=%s",
        test_size,
        random_state,
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_train, X_test, y_train, y_test


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Path:
    """Persist the train/test split so downstream tasks do not recompute it.

    Written via a temp file + atomic rename: a concurrent reader (e.g. the DAG's
    evaluate_models task, or a manually run `main.py`) must never see a partially
    written pickle.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / "train_test_split.pkl"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump((X_train, X_test, y_train, y_test), handle)
    tmp_path.replace(path)
    LOGGER.info("Saved train/test split to %s", path)
    return path


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the persisted train/test split."""
    path = PROCESSED_DATA_DIR / "train_test_split.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Train/test split not found at {path}. Run the prepare_data step first."
        )
    with path.open("rb") as handle:
        X_train, X_test, y_train, y_test = pickle.load(handle)
    return X_train, X_test, y_train, y_test
