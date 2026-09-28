"""MLflow experiment tracking helpers.

This is the only module that talks to MLflow.  Keeping it separate means the
Airflow DAG stays readable and the ML code stays testable without a tracking
server.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
)

LOGGER = logging.getLogger(__name__)

# scikit-learn tree ensembles (RandomForest, GradientBoosting) share this node
# storage type. skops will not deserialise it unless it is explicitly trusted.
TRUSTED_SKOPS_TYPES = ["sklearn.tree._tree.Tree"]


class MLflowLoggingError(RuntimeError):
    """Raised when an MLflow logging operation fails."""


def configure_tracking(tracking_uri: str | None = None) -> str:
    """Point every MLflow call at the local tracking server.

    Airflow tasks and the MLflow UI must use the same URI, otherwise runs would
    be written to a different store than the one being viewed.
    """
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    LOGGER.info("MLflow tracking URI set to %s", mlflow.get_tracking_uri())
    return mlflow.get_tracking_uri()


def get_or_create_experiment(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME, tracking_uri: str | None = None
) -> str:
    """Return the experiment id, creating the experiment if needed."""
    configure_tracking(tracking_uri)
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            LOGGER.info("Created MLflow experiment '%s' (id=%s)", experiment_name, experiment_id)
        else:
            experiment_id = experiment.experiment_id
            LOGGER.info(
                "Reusing MLflow experiment '%s' (id=%s, runs are never overwritten)",
                experiment_name,
                experiment_id,
            )
    except Exception as exc:
        raise MLflowLoggingError(
            f"Could not get or create MLflow experiment '{experiment_name}' at "
            f"{mlflow.get_tracking_uri()}: {exc}"
        ) from exc
    return str(experiment_id)


def log_run(
    model,
    model_name: str,
    model_display_name: str,
    params: dict[str, object],
    metrics: dict[str, float] | None = None,
    feature_names: list[str] | None = None,
    X_train: pd.DataFrame | None = None,
    y_test=None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Create one MLflow run per model: parameters, metrics, model, artifacts.

    Returns the run id so a later task (e.g. evaluation) can add metrics to the
    very same run.
    """
    experiment_id = get_or_create_experiment(experiment_name, tracking_uri)

    run_params = {"model_type": type(model).__name__, **params}
    if feature_names is not None:
        run_params["n_features"] = len(feature_names)

    try:
        with mlflow.start_run(
            run_name=model_display_name, experiment_id=experiment_id, tags={"model_key": model_name}
        ) as run:
            mlflow.log_params(run_params)

            if metrics:
                mlflow.log_metrics(metrics)

            input_example = None
            if X_train is not None:
                example = X_train.head(5)
                input_example = example.to_numpy()
                mlflow.log_dict(
                    {"feature_names": list(example.columns)},
                    "feature_names.json",
                )

            # `name` is the current MLflow API; `artifact_path` is removed in MLflow 3.
            # MLflow >= 3 serialises scikit-learn models with `skops`, which refuses
            # to load tree models until the shared node-storage type is declared as
            # trusted. The models are trained locally by this pipeline, so trusting
            # this single type is safe and necessary.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=input_example,
                skops_trusted_types=TRUSTED_SKOPS_TYPES,
            )

            mlflow.log_dict(
                {
                    "model": model_display_name,
                    "model_key": model_name,
                    "params": run_params,
                    "metrics": metrics or {},
                    "n_test_samples": int(len(y_test)) if y_test is not None else None,
                },
                "model_summary.json",
            )

            # MLflow 3 dropped `RunInfo.run_uri`, so the UI link is built here.
            run_url = (
                f"{mlflow.get_tracking_uri().rstrip('/')}"
                f"/#/experiments/{experiment_id}/runs/{run.info.run_id}"
            )
            LOGGER.info(
                "Logged MLflow run for %s -> run_id=%s url=%s",
                model_display_name,
                run.info.run_id,
                run_url,
            )
            return run.info.run_id
    except MLflowLoggingError:
        raise
    except Exception as exc:
        raise MLflowLoggingError(
            f"Failed to log model '{model_name}' to MLflow at "
            f"{mlflow.get_tracking_uri()}: {exc}"
        ) from exc


def log_evaluation(
    run_id: str,
    model_name: str,
    model_display_name: str,
    metrics: dict[str, float],
    plot_path: Path | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Add metrics (and optionally the plot) to an existing MLflow run."""
    configure_tracking(tracking_uri)
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            if plot_path is not None and Path(plot_path).exists():
                mlflow.log_artifact(str(plot_path), artifact_path="plots")
        LOGGER.info(
            "Logged metrics %s to run_id=%s (%s)", metrics, run_id, model_display_name
        )
        return run_id
    except Exception as exc:
        raise MLflowLoggingError(
            f"Failed to log evaluation for '{model_name}' to run_id={run_id}: {exc}"
        ) from exc


def write_comparison_report(rows: list[dict[str, object]]) -> Path:
    """Write the model comparison table to ``artifacts/reports/``."""
    import pandas as pd

    if not rows:
        raise MLflowLoggingError("Cannot write a comparison report with zero model results.")

    frame = pd.DataFrame(rows)
    frame = frame[["model", "mae", "rmse", "r2"]]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS_DIR / "model_comparison.csv"
    frame.to_csv(csv_path, index=False)

    json_path = REPORTS_DIR / "model_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    LOGGER.info("Wrote model comparison to %s and %s", csv_path, json_path)
    return csv_path
