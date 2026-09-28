"""Regression metrics and diagnostic plotting."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when predictions cannot be scored."""


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE and R-squared for a regression model.

    Full float precision is preserved; values are not rounded.
    """
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()

    if y_true.size == 0:
        raise EvaluationError("Cannot evaluate on an empty target array.")
    if y_true.shape != y_pred.shape:
        raise EvaluationError(
            f"Shape mismatch between y_true {y_true.shape} and y_pred {y_pred.shape}."
        )

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # root_mean_squared_error exists only in newer scikit-learn; the manual
        # form keeps this working across versions.
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    LOGGER.info("Metrics computed: %s", metrics)
    return metrics


def plot_actual_vs_predicted(
    y_true, y_pred, model_label: str, output_path: Path, max_points: int = 1000
) -> Path:
    """Save an actual-vs-predicted scatter plot and return its path."""
    import matplotlib

    matplotlib.use("Agg")  # headless: Airflow has no display
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()

    if y_true.size > max_points:  # keep the plot readable and the file small
        step = y_true.size // max_points + 1
        y_true_plot, y_pred_plot = y_true[::step], y_pred[::step]
    else:
        y_true_plot, y_pred_plot = y_true, y_pred

    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.4, s=14, edgecolors="none")
    ax.plot([low, high], [low, high], linestyle="--", color="crimson", label="Perfect prediction")
    ax.set_xlabel("Actual house value (MedHouseVal, $100k units)")
    ax.set_ylabel("Predicted house value")
    ax.set_title(f"Actual vs Predicted - {model_label}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved actual-vs-predicted plot to %s", output_path)
    return output_path
