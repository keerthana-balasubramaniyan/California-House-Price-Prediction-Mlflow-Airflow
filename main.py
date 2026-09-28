"""Command-line entry point that runs the whole pipeline without Airflow.

Useful for development and for verifying the ML code in isolation:

    python main.py                      # MLflow tracking URI from config/.env
    python main.py --no-mlflow          # skip tracking entirely
    python main.py --local-mlflow       # write runs to ./mlruns instead of the server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data, evaluate, tracking  # noqa: E402
from src.config import MLFLOW_TRACKING_URI, RANDOM_STATE, REPORTS_DIR, TEST_SIZE  # noqa: E402
from src.train import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    build_model,
    get_model_params,
    save_model,
    train_model,
)

MODEL_ORDER = ["random_forest", "gradient_boosting"]


def main() -> int:
    parser = argparse.ArgumentParser(description="California House Price ML pipeline")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Do not log anything to MLflow"
    )
    parser.add_argument(
        "--local-mlflow",
        action="store_true",
        help="Log to a local ./mlruns directory instead of the tracking server",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    tracking_uri = MLFLOW_TRACKING_URI
    if args.local_mlflow:
        tracking_uri = f"file://{(Path(__file__).resolve().parent / 'mlruns').as_posix()}"
    use_mlflow = not args.no_mlflow

    # 1. Load -------------------------------------------------------------
    X, y = data.load_dataset()

    # 2. Validate ---------------------------------------------------------
    report = data.validate_dataset(X, y)
    print("\n=== DATASET ===")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_names"}, indent=2))
    print("feature_names:", report["feature_names"])
    print(X.describe().to_string(), "\n")

    # 3. Preprocess / split ----------------------------------------------
    X_train, X_test, y_train, y_test = data.prepare_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    data.save_split(X_train, X_test, y_train, y_test)

    # 4-5. Train + evaluate ----------------------------------------------
    results = []
    for model_name in MODEL_ORDER:
        display = MODEL_DISPLAY_NAMES[model_name]
        params = get_model_params(model_name)
        print(f"\n=== {display} ===")

        model = build_model(model_name, params)
        train_model(model, X_train, y_train)
        save_model(model, model_name)

        predictions = model.predict(X_test)
        metrics = evaluate.compute_metrics(y_test, predictions)
        print(f"{display}: " + "  ".join(f"{k}={v:.6f}" for k, v in metrics.items()))

        plot_path = REPORTS_DIR / f"{model_name}_actual_vs_predicted.png"
        evaluate.plot_actual_vs_predicted(y_test, predictions, display, plot_path)

        run_id = None
        if use_mlflow:
            # Training logs params + model, evaluation logs metrics + plot to
            # the SAME run, mirroring the two Airflow tasks.
            run_id = tracking.log_run(
                model=model,
                model_name=model_name,
                model_display_name=display,
                params=params,
                feature_names=list(X_train.columns),
                X_train=X_train,
                y_test=y_test,
                tracking_uri=tracking_uri,
            )
            tracking.log_evaluation(
                run_id=run_id,
                model_name=model_name,
                model_display_name=display,
                metrics=metrics,
                plot_path=plot_path,
                tracking_uri=tracking_uri,
            )
            print(f"MLflow run_id={run_id} uri={tracking_uri}")

        results.append(
            {
                "model": display,
                "model_key": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "run_id": run_id,
            }
        )

    # 6. Comparison report -------------------------------------------------
    comparison = tracking.write_comparison_report(results) if use_mlflow else None
    table = pd.DataFrame(results)[["model", "mae", "rmse", "r2"]]
    print("\n=== MODEL COMPARISON ===")
    print(table.to_string(index=False))
    if comparison:
        print(f"\nReport written to {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
