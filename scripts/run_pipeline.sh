#!/usr/bin/env bash
# Run the ML pipeline directly (no Airflow). Useful for quick iteration and
# for proving the ML code works on its own.
#
# Usage:
#   bash scripts/run_pipeline.sh              # log to the MLflow server
#   bash scripts/run_pipeline.sh --no-mlflow  # no tracking at all
#   bash scripts/run_pipeline.sh --local-mlflow   # track into ./mlruns
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/houseprice}"
export PATH="$VENV_DIR/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

cd "$PROJECT_DIR"
exec python main.py "$@"
