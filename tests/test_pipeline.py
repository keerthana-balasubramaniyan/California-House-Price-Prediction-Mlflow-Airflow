"""Tests for the California House Price pipeline.

Covers the dataset, validation, preprocessing, model training, metrics,
MLflow tracking and DAG importability. Tracking tests use a local sqlite
tracking URI (same backend kind the project's own MLflow server uses), so no
tracking server is required. MLflow >= 3 put the plain file store in
maintenance mode and rejects it outright, hence sqlite here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data as data_module  # noqa: E402
from src import evaluate as evaluate_module  # noqa: E402
from src import tracking as tracking_module  # noqa: E402
from src import train as train_module  # noqa: E402
from src.config import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE  # noqa: E402

MODEL_KEYS = ["random_forest", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset():
    return data_module.load_dataset()


@pytest.fixture(scope="module")
def split(dataset):
    X, y = dataset
    return data_module.prepare_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def fitted_models(split):
    """Train both models once on a small sample; shared across model tests."""
    X_train, _, y_train, _ = split
    sample = slice(0, 800)
    fitted = {}
    for key in MODEL_KEYS:
        model = train_module.build_model(key)
        train_module.train_model(model, X_train.iloc[sample], y_train.iloc[sample])
        fitted[key] = model
    return fitted


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_dataset_loads(dataset):
    features, target = dataset
    assert features is not None
    assert target is not None
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)


def test_dataset_is_not_empty(dataset):
    features, target = dataset
    assert not features.empty
    assert not target.empty
    # The published California Housing dataset has 20,640 rows and 8 features.
    assert features.shape == (20640, 8)
    assert target.shape[0] == 20640


def test_expected_features_exist(dataset):
    features, _ = dataset
    for column in FEATURE_COLUMNS:
        assert column in features.columns, f"missing feature: {column}"


def test_target_is_numeric_regression_target(dataset):
    _, target = dataset
    assert pd.api.types.is_numeric_dtype(target)
    assert target.min() > 0
    assert target.max() < 10  # California housing target is capped at 5.00001


def test_no_missing_values(dataset):
    features, target = dataset
    assert features.isna().sum().sum() == 0
    assert target.isna().sum() == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_dataset_passes(dataset):
    features, target = dataset
    report = data_module.validate_dataset(features, target)
    assert report["status"] == "passed"
    assert report["n_samples"] == 20640
    assert report["n_features"] == 8
    assert report["n_missing_values"] == 0


def test_validate_dataset_rejects_empty():
    empty_X = pd.DataFrame(columns=FEATURE_COLUMNS)
    empty_y = pd.Series(dtype="float64")
    with pytest.raises(data_module.DataValidationError):
        data_module.validate_dataset(empty_X, empty_y)


def test_validate_dataset_rejects_missing_column(dataset):
    features, target = dataset
    with pytest.raises(data_module.DataValidationError, match="Missing expected feature"):
        data_module.validate_dataset(features.drop(columns=["MedInc"]), target)


def test_validate_dataset_rejects_length_mismatch(dataset):
    features, target = dataset
    with pytest.raises(data_module.DataValidationError, match="Row count mismatch"):
        data_module.validate_dataset(features, target.iloc[:-5])


def test_validate_dataset_rejects_missing_values(dataset):
    features, target = dataset
    broken = features.copy()
    broken.loc[0, "MedInc"] = np.nan
    with pytest.raises(data_module.DataValidationError, match="missing value"):
        data_module.validate_dataset(broken, target)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_split_shapes_are_correct(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) + len(X_test) == 20640
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == 8 == X_test.shape[1]
    # 80/20 split
    assert len(X_test) == pytest.approx(20640 * TEST_SIZE, rel=0.01)


def test_split_has_no_overlap(split):
    """Test rows must never appear in the training set (no data leakage)."""
    X_train, X_test, _, _ = split
    train_ids = set(X_train.index)
    test_ids = set(X_test.index)
    assert train_ids.isdisjoint(test_ids)


def test_split_is_reproducible(dataset):
    X, y = dataset
    a = data_module.prepare_data(X, y, random_state=42)
    b = data_module.prepare_data(X, y, random_state=42)
    assert a[0].equals(b[0])
    assert a[2].equals(b[2])


def test_split_uses_fixed_random_state(dataset):
    X, y = dataset
    a = data_module.prepare_data(X, y, random_state=42)
    c = data_module.prepare_data(X, y, random_state=7)
    assert not a[1].equals(c[1])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_model_builds_with_expected_class(model_key):
    model = train_module.build_model(model_key)
    expected = {
        "random_forest": "RandomForestRegressor",
        "gradient_boosting": "GradientBoostingRegressor",
    }[model_key]
    assert type(model).__name__ == expected
    assert getattr(model, "random_state") == RANDOM_STATE


def test_build_model_rejects_unknown_model():
    with pytest.raises(train_module.ModelTrainingError):
        train_module.build_model("does_not_exist")


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_model_trains_and_predicts(model_key, fitted_models, split):
    _, X_test, _, _ = split
    model = fitted_models[model_key]
    predictions = model.predict(X_test.iloc[:100])
    assert predictions.shape == (100,)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_model_training_is_deterministic(model_key, split):
    X_train, _, y_train, _ = split
    sample = slice(0, 500)
    a = train_module.build_model(model_key).fit(X_train.iloc[sample], y_train.iloc[sample])
    b = train_module.build_model(model_key).fit(X_train.iloc[sample], y_train.iloc[sample])
    assert np.array_equal(a.predict(X_train.iloc[sample]), b.predict(X_train.iloc[sample]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metrics_are_valid():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.7])
    metrics = evaluate_module.compute_metrics(y_true, y_pred)
    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    # RMSE is always >= MAE for the same errors
    assert metrics["rmse"] >= metrics["mae"]
    assert -np.inf < metrics["r2"] <= 1.0


def test_perfect_predictions_give_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_module.compute_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_metrics_match_sklearn_values():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rng = np.random.default_rng(42)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.5, size=200)
    metrics = evaluate_module.compute_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert metrics["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y_true, y_pred)))
    assert metrics["r2"] == pytest.approx(r2_score(y_true, y_pred))


def test_metrics_keep_full_precision():
    # MAE averages the two absolute errors: (0.2345678901234 + 0) / 2
    metrics = evaluate_module.compute_metrics([1.0, 2.0], [1.2345678901234, 2.0])
    assert metrics["mae"] == pytest.approx(0.1172839450617, rel=1e-12)


def test_metrics_reject_shape_mismatch():
    with pytest.raises(evaluate_module.EvaluationError):
        evaluate_module.compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0])


def test_metrics_reject_empty_input():
    with pytest.raises(evaluate_module.EvaluationError):
        evaluate_module.compute_metrics([], [])


def test_plot_is_created(tmp_path):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    out = evaluate_module.plot_actual_vs_predicted(y, y + 0.1, "Test Model", tmp_path / "p.png")
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# MLflow tracking (local file store, no server required)
# ---------------------------------------------------------------------------
@pytest.fixture()
def local_tracking_uri(tmp_path):
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_log_run_then_log_evaluation_round_trip(local_tracking_uri, split):
    """Mirrors the two-step DAG contract: train_models creates the run and
    logs params+model, evaluate_models re-opens the SAME run_id to add
    metrics+plot. A rename of either side's dict keys would break this at
    DAG runtime with no test catching it - this test pins those keys.
    """
    import mlflow

    X_train, X_test, y_train, y_test = split
    model_key = "random_forest"
    model = train_module.build_model(model_key)
    train_module.train_model(model, X_train.iloc[:200], y_train.iloc[:200])

    experiment_name = f"test-{Path(local_tracking_uri).name}"
    run_id = tracking_module.log_run(
        model=model,
        model_name=model_key,
        model_display_name="Random Forest Regressor",
        params=train_module.get_model_params(model_key),
        feature_names=list(X_train.columns),
        X_train=X_train.iloc[:200],
        experiment_name=experiment_name,
        tracking_uri=local_tracking_uri,
    )
    assert run_id

    predictions = model.predict(X_test.iloc[:100])
    metrics = evaluate_module.compute_metrics(y_test.iloc[:100], predictions)
    returned_run_id = tracking_module.log_evaluation(
        run_id=run_id,
        model_name=model_key,
        model_display_name="Random Forest Regressor",
        metrics=metrics,
        tracking_uri=local_tracking_uri,
    )
    assert returned_run_id == run_id

    mlflow.set_tracking_uri(local_tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["model_type"] == "RandomForestRegressor"
    for key in ("mae", "rmse", "r2"):
        assert key in run.data.metrics


def test_get_or_create_experiment_reuses_existing(local_tracking_uri):
    first_id = tracking_module.get_or_create_experiment(
        "reuse-test", tracking_uri=local_tracking_uri
    )
    second_id = tracking_module.get_or_create_experiment(
        "reuse-test", tracking_uri=local_tracking_uri
    )
    assert first_id == second_id


def test_write_comparison_report_rejects_empty_rows():
    with pytest.raises(tracking_module.MLflowLoggingError):
        tracking_module.write_comparison_report([])


# ---------------------------------------------------------------------------
# Airflow DAG
# ---------------------------------------------------------------------------
def test_dag_file_imports_successfully():
    """The DAG must be importable without a running Airflow scheduler."""
    dag_path = PROJECT_ROOT / "dags" / "house_price_pipeline.py"
    assert dag_path.exists(), f"missing DAG file: {dag_path}"

    spec = importlib.util.spec_from_file_location("house_price_pipeline", dag_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from airflow.sdk import DAG

    dag_obj = module.house_price_pipeline_dag
    assert isinstance(dag_obj, DAG)
    assert dag_obj.dag_id == "house_price_pipeline"
    assert set(dag_obj.task_ids) == {
        "load_data",
        "validate_data",
        "prepare_data",
        "train_models",
        "evaluate_models",
    }


def test_dag_task_dependencies_are_linear():
    dag_path = PROJECT_ROOT / "dags" / "house_price_pipeline.py"
    spec = importlib.util.spec_from_file_location("house_price_pipeline_deps", dag_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dag_obj = module.house_price_pipeline_dag
    expected = {
        "load_data": set(),
        "validate_data": {"load_data"},
        "prepare_data": {"validate_data"},
        "train_models": {"prepare_data"},
        "evaluate_models": {"train_models"},
    }
    for task_id, upstream in expected.items():
        assert dag_obj.task_dict[task_id].upstream_task_ids == upstream
