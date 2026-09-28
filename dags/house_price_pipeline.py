"""Airflow DAG for the California House Price prediction pipeline.

Airflow owns the orchestration (task order, dependencies, retries, logs, UI).
MLflow owns experiment tracking. All machine-learning logic lives in ``src/``.

Task flow
---------
    load_data -> validate_data -> prepare_data -> train_models -> evaluate_models
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Airflow runs DAG files from the `dags/` folder, so the project root (parent of
# this folder) has to be importable for `src` and `main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Guarantee that the Airflow workers and the MLflow UI talk to the same store.
# An explicit MLFLOW_TRACKING_URI in the environment still wins.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

import pendulum  # noqa: E402
from airflow.sdk import dag, task  # noqa: E402

from src import data as data_module  # noqa: E402
from src import evaluate as evaluate_module  # noqa: E402
from src import tracking as tracking_module  # noqa: E402
from src import train as train_module  # noqa: E402
from src.config import MLFLOW_EXPERIMENT_NAME, RANDOM_STATE, REPORTS_DIR, TEST_SIZE  # noqa: E402

LOGGER = logging.getLogger(__name__)

MODEL_ORDER = ["random_forest", "gradient_boosting"]

DEFAULT_ARGS = {
    "owner": "mlops-learning",
    "retries": 1,
    "retry_delay": pendulum.duration(seconds=30),
}


@dag(
    dag_id="house_price_pipeline",
    description="California House Price regression pipeline orchestrated by Airflow and tracked in MLflow",
    schedule="@daily",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["mlops", "airflow", "mlflow", "regression", "california-housing"],
)
def house_price_pipeline():
    @task
    def load_data() -> dict:
        """Load the California Housing dataset into a pandas DataFrame."""
        LOGGER.info("Loading California Housing dataset...")
        features, target = data_module.load_dataset()
        LOGGER.info("Loaded %d rows x %d features", features.shape[0], features.shape[1])
        return {
            "n_samples": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "feature_names": list(features.columns),
            "target_name": str(target.name),
        }

    @task
    def validate_data(dataset_info: dict) -> dict:
        """Fail fast if the dataset does not match the expected structure."""
        LOGGER.info("Validating dataset: %s", dataset_info)
        features, target = data_module.load_dataset()
        report = data_module.validate_dataset(features, target)

        report_path = REPORTS_DIR / "dataset_validation.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        LOGGER.info("Validation report written to %s", report_path)
        return report

    @task
    def prepare_data(validation_report: dict) -> dict:
        """Split the dataset into train and test sets (no data leakage)."""
        if validation_report.get("status") != "passed":
            raise ValueError(
                f"Refusing to prepare data: validation status is "
                f"'{validation_report.get('status')}'"
            )
        LOGGER.info("Splitting dataset (test_size=%s, random_state=%s)", TEST_SIZE, RANDOM_STATE)
        features, target = data_module.load_dataset()
        X_train, X_test, y_train, y_test = data_module.prepare_data(
            features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        path = data_module.save_split(X_train, X_test, y_train, y_test)
        return {
            "split_path": str(path),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
        }

    @task
    def train_models(split_info: dict) -> list[dict]:
        """Train each model and log parameters + model to a separate MLflow run."""
        X_train, X_test, y_train, _ = data_module.load_split()
        LOGGER.info("Training %d models on %d rows", len(MODEL_ORDER), X_train.shape[0])

        experiment_id = tracking_module.get_or_create_experiment(MLFLOW_EXPERIMENT_NAME)
        LOGGER.info("Using MLflow experiment id=%s (%s)", experiment_id, MLFLOW_EXPERIMENT_NAME)

        trained: list[dict] = []
        for model_name in MODEL_ORDER:
            display_name = train_module.MODEL_DISPLAY_NAMES[model_name]
            params = train_module.get_model_params(model_name)

            LOGGER.info("Training %s with %s", display_name, params)
            model = train_module.build_model(model_name, params)
            train_module.train_model(model, X_train, y_train)
            train_module.save_model(model, model_name)

            run_id = tracking_module.log_run(
                model=model,
                model_name=model_name,
                model_display_name=display_name,
                params=params,
                feature_names=list(X_train.columns),
                X_train=X_train,
                y_test=None,
                experiment_name=MLFLOW_EXPERIMENT_NAME,
            )
            LOGGER.info("Logged %s to MLflow run_id=%s", display_name, run_id)
            trained.append(
                {
                    "model_key": model_name,
                    "model": display_name,
                    "run_id": run_id,
                }
            )
        return trained

    @task
    def evaluate_models(trained: list[dict]) -> list[dict]:
        """Score each model and add metrics + plot to its existing MLflow run."""
        _, X_test, _, y_test = data_module.load_split()
        LOGGER.info("Evaluating %d models on %d test rows", len(trained), X_test.shape[0])

        results: list[dict] = []
        for entry in trained:
            model_name = entry["model_key"]
            display_name = entry["model"]
            run_id = entry["run_id"]

            model = train_module.load_model(model_name)
            predictions = model.predict(X_test)
            metrics = evaluate_module.compute_metrics(y_test, predictions)
            LOGGER.info("%s metrics: %s", display_name, metrics)

            plot_path = REPORTS_DIR / f"{model_name}_actual_vs_predicted.png"
            evaluate_module.plot_actual_vs_predicted(
                y_test, predictions, display_name, plot_path
            )

            tracking_module.log_evaluation(
                run_id=run_id,
                model_name=model_name,
                model_display_name=display_name,
                metrics=metrics,
                plot_path=plot_path,
            )

            results.append({"model": display_name, **metrics, "run_id": run_id})

        report_path = tracking_module.write_comparison_report(results)
        LOGGER.info("Model comparison written to %s", report_path)
        for row in results:
            LOGGER.info(
                "%s -> MAE=%.6f RMSE=%.6f R2=%.6f (run_id=%s)",
                row["model"], row["mae"], row["rmse"], row["r2"], row["run_id"],
            )
        return results

    load = load_data()
    validated = validate_data(load)
    prepared = prepare_data(validated)
    trained = train_models(prepared)
    evaluate_models(trained)


# Airflow 3's `@dag` decorator returns a factory function rather than the DAG
# itself, so the DAG object is produced by calling it. The object is kept in a
# module-level name so that tests and tooling can inspect it after import.
house_price_pipeline_dag = house_price_pipeline()
