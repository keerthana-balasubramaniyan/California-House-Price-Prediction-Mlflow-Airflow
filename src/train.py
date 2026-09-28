"""Model construction and training.

Pure scikit-learn code: no Airflow and no MLflow imports here, so the training
logic can be unit-tested on its own.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.config import MODEL_PARAMS, MODELS_DIR, RANDOM_STATE

LOGGER = logging.getLogger(__name__)

MODEL_CLASSES = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
}


class ModelTrainingError(RuntimeError):
    """Raised when a model cannot be built or trained."""


def build_model(model_name: str, params: dict[str, object] | None = None):
    """Instantiate a regressor from the project configuration."""
    if model_name not in MODEL_CLASSES:
        raise ModelTrainingError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_CLASSES)}"
        )

    model_params = dict(MODEL_PARAMS[model_name] if params is None else params)
    model_params.setdefault("random_state", RANDOM_STATE)

    LOGGER.info("Building %s with params: %s", model_name, model_params)
    return MODEL_CLASSES[model_name](**model_params)


def get_model_params(model_name: str) -> dict[str, object]:
    """Return the hyper-parameters used for a model."""
    if model_name not in MODEL_PARAMS:
        raise ModelTrainingError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_PARAMS)}"
        )
    return dict(MODEL_PARAMS[model_name])


def train_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a model, raising a clear error on failure."""
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        raise ModelTrainingError(
            f"Training failed for {type(model).__name__}: {exc}"
        ) from exc
    LOGGER.info("Trained %s", type(model).__name__)
    return model


def save_model(model, model_name: str) -> Path:
    """Persist a trained model to ``models/``.

    Written via a temp file + atomic rename so a concurrent reader (the DAG and
    a manually run `main.py` both write to the same path) never sees a
    partially written pickle.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_name}.pkl"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(model, handle)
    tmp_path.replace(path)
    LOGGER.info("Saved model %s -> %s", model_name, path)
    return path


def load_model(model_name: str):
    """Load a previously trained model from ``models/``."""
    path = MODELS_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found at {path}.")
    with path.open("rb") as handle:
        return pickle.load(handle)
