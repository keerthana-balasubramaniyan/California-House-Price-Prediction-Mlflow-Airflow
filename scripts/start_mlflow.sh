#!/usr/bin/env bash
# Start the MLflow tracking server (UI + REST API) on http://localhost:5000
#
# Usage:  bash scripts/start_mlflow.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/houseprice}"
export PATH="$VENV_DIR/bin:$PATH"

cd "$PROJECT_DIR"
mkdir -p mlflow/artifacts logs

# Scoped to this project's backend store, not just "mlflow server" - a
# machine can run other MLflow projects on other ports at the same time.
if pgrep -f "mlflow server .*$PROJECT_DIR/mlflow/mlflow.db" >/dev/null 2>&1; then
  echo "MLflow server already running. Stop it with: bash scripts/stop_mlflow.sh"
  curl -s -o /dev/null -w "health check: HTTP %{http_code}\n" http://127.0.0.1:5000/health
  exit 0
fi

echo "Starting MLflow tracking server on http://127.0.0.1:5000 ..."
nohup setsid mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "sqlite:///$PROJECT_DIR/mlflow/mlflow.db" \
  --default-artifact-root "$PROJECT_DIR/mlflow/artifacts" \
  > logs/mlflow_server.log 2>&1 &
disown

for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:5000/health 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    echo "MLflow is ready (after ${i}s). Open http://localhost:5000"
    exit 0
  fi
  sleep 1
done

echo "ERROR: MLflow did not become healthy. Last log lines:"
tail -20 logs/mlflow_server.log
exit 1
