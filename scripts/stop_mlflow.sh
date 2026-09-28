#!/usr/bin/env bash
# Stop the MLflow tracking server for THIS project.
#
# Always stop the server BEFORE deleting mlflow/mlflow.db. Removing the SQLite
# file while the server has it open makes the server return HTTP 503.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Matched by this project's backend store path, not just "mlflow server" - a
# machine can run other MLflow projects on other ports at the same time.
PATTERN="mlflow server .*$PROJECT_DIR/mlflow/mlflow.db"

if pgrep -f "$PATTERN" >/dev/null 2>&1; then
  pkill -f "$PATTERN" && echo "sent TERM to mlflow server"
  for _ in $(seq 1 15); do pgrep -f "$PATTERN" >/dev/null || break; sleep 1; done
  pkill -9 -f "$PATTERN" 2>/dev/null
  echo "MLflow server stopped."
else
  echo "MLflow server is not running."
fi
