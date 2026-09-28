#!/usr/bin/env bash
# Start Apache Airflow in standalone mode (API server + scheduler +
# DAG processor + triggerer) with the UI on http://localhost:8080
#
# Usage:  bash scripts/start_airflow.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/houseprice}"
export PATH="$VENV_DIR/bin:$PATH"

export AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_DIR/dags"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

cd "$PROJECT_DIR"
mkdir -p logs

# curl exits 0 on any HTTP response, including 503, so check the status code.
health_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
  http://127.0.0.1:8080/api/v2/monitor/health 2>/dev/null || true)
if [ "$health_code" = "200" ]; then
  echo "Airflow already running. Open http://localhost:8080"
  exit 0
fi

echo "Initialising Airflow metadata database (if needed)..."
airflow db migrate > logs/airflow_db_migrate.log 2>&1 || {
  echo "ERROR: db migrate failed"; tail -20 logs/airflow_db_migrate.log; exit 1;
}

echo "Starting Airflow standalone ..."
nohup setsid airflow standalone > logs/airflow_standalone.log 2>&1 &
disown

for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8080/api/v2/monitor/health 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    echo "Airflow is ready (after ${i}s). Open http://localhost:8080"
    echo
    echo "Login with user 'admin'. The generated password is printed in:"
    echo "  logs/airflow_standalone.log   (search for \"Password for user 'admin'\")"
    echo "  or read: $AIRFLOW_HOME/simple_auth_manager_passwords.json.generated"
    exit 0
  fi
  sleep 1
done

echo "ERROR: Airflow did not become healthy. Last log lines:"
tail -30 logs/airflow_standalone.log
exit 1
