#!/usr/bin/env bash
# Stop the Airflow standalone stack.
#
# `airflow standalone` forks child processes (api-server, scheduler, dag-processor,
# triggerer, workers, log servers). Killing only the parent leaves those children
# running and still holding port 8080, so every component is matched here.
set -uo pipefail

PATTERNS=(
  "airflow standalone"
  "airflow api_server"
  "airflow scheduler"
  "airflow triggerer"
  "airflow dag_processor"
  "airflow worker"
  "airflow serve-logs"
)

found=0
for pattern in "${PATTERNS[@]}"; do
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    pkill -f "$pattern" 2>/dev/null && found=1
  fi
done

[ "$found" = "1" ] || { echo "Airflow is not running."; exit 0; }
echo "sent TERM to Airflow components; waiting for exit..."

for _ in $(seq 1 20); do
  still=0
  for pattern in "${PATTERNS[@]}"; do
    pgrep -f "$pattern" >/dev/null 2>&1 && still=1
  done
  [ "$still" = "0" ] && break
  sleep 1
done

for pattern in "${PATTERNS[@]}"; do
  pkill -9 -f "$pattern" 2>/dev/null
done
sleep 1

remaining=0
for pattern in "${PATTERNS[@]}"; do
  pgrep -f "$pattern" >/dev/null 2>&1 && remaining=1
done

if [ "$remaining" = "1" ]; then
  echo "WARNING: some Airflow processes are still running."
  pgrep -af "airflow" | head -10
  exit 1
fi

echo "Airflow stopped (all components)."
