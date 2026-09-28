# California House Price Prediction with Airflow and MLflow

A small, complete MLOps project that predicts California house prices.
**Apache Airflow** orchestrates the workflow; **MLflow** tracks every experiment,
parameter, metric, model and artifact.

---

## 1. Project Overview

This project trains and compares two scikit-learn regressors on the
**California Housing** dataset and records the results in MLflow.

| | |
|---|---|
| Problem type | Regression (predict a continuous house-value target) |
| Models compared | `RandomForestRegressor`, `GradientBoostingRegressor` |
| Metrics | MAE, RMSE, R² |
| Orchestrator | Apache Airflow 3.3.2 (`house_price_pipeline` DAG) |
| Tracker | MLflow 3.16.1 (experiment `california-house-price-prediction`) |
| Airflow UI | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |

---

## 2. Objective

The point of this project is to show **how the two tools divide the work**:

* **Airflow** answers *when* and *in what order* things run. It owns the task
  graph, dependencies, scheduling, retries, status and logs.
* **MLflow** answers *what happened*. It owns the experiment, the runs, and the
  parameters / metrics / artifacts each run produced.
* **Scikit-learn** answers *how the maths is done*. Loading, splitting, fitting,
  predicting and scoring all live in `src/`.

Airflow is deliberately kept thin: the DAG wires tasks together and delegates
all real logic to `src/`. Every trigger of the DAG adds new runs to the same
MLflow experiment, so you can compare executions side by side.

---

## 3. Dataset

The dataset comes from **scikit-learn** — there is no CSV to download:

```python
from sklearn.datasets import fetch_california_housing
```

It is derived from the 1990 California census. The task is to predict the
median house value for California districts.

| Property | Value (verified by running the pipeline) |
|---|---|
| Samples | **20,640** |
| Features | **8** |
| Target | `MedHouseVal` (named `TARGET_COLUMN` in this project) |
| Missing values | **0** |
| Target range | 0.14999 – 5.00001 (in units of $100,000) |
| Target mean | 2.0686 |

Features:

| Feature | Meaning |
|---|---|
| `MedInc` | Median income of households in the district |
| `HouseAge` | Median age of the houses |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | District population |
| `AveOccup` | Average number of household members |
| `Latitude` | Latitude, higher = further north |
| `Longitude` | Longitude, higher = further west |

`src/data.py` inspects and caches the dataset at `data/raw/california_housing.pkl`,
so repeat runs are fast and work offline.

---

## 4. Architecture

```text
                              AIRFLOW
                    (task graph, schedule, retries, logs)
                                 |
                                 v
  +----------------------------------------------------------------+
  |  load_data                                                      |
  |      fetch_california_housing()  ->  data/raw/*.pkl             |
  |              |                                                   |
  |              v                                                   |
  |  validate_data                                                   |
  |      non-empty? columns present? lengths match?                  |
  |      any NaN? numeric dtypes?   -> artifacts/reports/           |
  |              |                     dataset_validation.json       |
  |              v                                                   |
  |  prepare_data                                                    |
  |      train_test_split(test_size=0.2, random_state=42)           |
  |      -> data/processed/train_test_split.pkl                     |
  |              |                                                   |
  |              v                                                   |
  |  train_models                                                    |
  |      +---------------------------+                              |
  |      | RandomForestRegressor     |                              |
  |      | GradientBoostingRegressor |                              |
  |      +-------------+-------------+                              |
  |                    |                                            |
  |                    v                                            |
  |  evaluate_models                                                 |
  |      MAE / RMSE / R2  +  actual-vs-predicted plot               |
  |                    |                                            |
  +--------------------|--------------------------------------------+
                       v
                     MLFLOW              (tracking URI:
                    http://127.0.0.1:5000   same for Airflow tasks
                                            and the browser UI)
              +-------------+-------------+
              |             |             |
              v             v             v
         Parameters      Metrics      Artifacts
        (model_type,     (mae,        (model, feature_names.json,
         n_estimators,    rmse,        model_summary.json,
         max_depth,       r2)          plots/*.png)
         learning_rate,
         random_state)
```

Data flows between tasks through **files on disk** (`data/`, `models/`,
`artifacts/`) and small JSON summaries through XCom. Large DataFrames are never
pushed through XCom.

---

## 5. Technologies

| Tool | Version | Role |
|---|---|---|
| Python | 3.12.3 | Language |
| Apache Airflow | 3.3.2 | Workflow orchestration |
| MLflow | 3.16.1 | Experiment tracking |
| Scikit-learn | 1.9.1 | Machine learning |
| Pandas | 3.0.5 | DataFrames |
| NumPy | 2.5.3 | Numeric arrays |
| Matplotlib | 3.11.2 | Diagnostic plots |
| Pytest | 9.1.1 | Tests |

---

## 6. Responsibilities

```text
Airflow        -> Workflow orchestration
MLflow         -> Experiment tracking
Scikit-learn   -> Machine learning
Pandas/NumPy   -> Data handling
Pytest         -> Testing
```

* `dags/house_price_pipeline.py` contains **no ML logic** — only task wiring.
* `src/data.py` — load, validate, split.
* `src/train.py` — build and fit models, save/load with pickle.
* `src/evaluate.py` — metrics and the actual-vs-predicted plot.
* `src/tracking.py` — the only module that imports MLflow.
* `src/config.py` — paths, seeds, hyper-parameters, tracking URI.
* `main.py` — runs the same pipeline without Airflow (useful for debugging).

---

## 7. Project Structure

```text
California House Price Prediction/
│
├── dags/
│   └── house_price_pipeline.py     # Airflow DAG (orchestration only)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # paths, seeds, model params, MLflow URI
│   ├── data.py                     # load + validate + split
│   ├── train.py                    # model building and training
│   ├── evaluate.py                 # MAE / RMSE / R2 + plot
│   └── tracking.py                 # MLflow logging
│
├── tests/
│   └── test_pipeline.py            # 33 pytest tests
│
├── scripts/
│   ├── start_mlflow.sh
│   ├── stop_mlflow.sh
│   ├── start_airflow.sh
│   ├── stop_airflow.sh
│   └── run_pipeline.sh             # run the ML pipeline without Airflow
│
├── data/                           # generated: raw/ + processed/
├── models/                         # generated: *.pkl
├── artifacts/                      # generated: reports, plots
├── mlflow/                         # generated: MLflow db + artifacts
├── logs/                           # generated: server logs
│
├── main.py
├── pytest.ini
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`data/`, `models/`, `artifacts/`, `mlflow/` and `logs/` hold generated output and
are git-ignored; each is kept in git only via a `.gitkeep` file.

---

## 8. Installation

### Important: Airflow must run in WSL2 (or Linux)

**Apache Airflow 3.x does not run on native Windows.** On import it calls
`os.register_at_fork()`, which only exists on POSIX systems, and the CLI fails
with:

```text
AttributeError: module 'os' has no attribute 'register_at_fork'
```

Airflow's own warning recommends WSL2, so this project is designed to run there.
Your Windows folders are still visible inside WSL under `/mnt/<drive>`.

Open **Ubuntu** (WSL2) and follow the commands below. All paths in this section
assume the project is at `/mnt/d/EA Sports/California House Price Prediction`.

```bash
# 1. Go to the project
cd "/mnt/d/EA Sports/California House Price Prediction"

# 2. Create the virtual environment.
#    NOTE: if `python3 -m venv` reports that python3-venv is missing, bootstrap
#    pip manually (needs no root):
python3 -m venv --without-pip ~/.venvs/houseprice
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/houseprice/bin/python /tmp/get-pip.py
~/.venvs/houseprice/bin/python -m pip install --upgrade pip
export PATH="$HOME/.venvs/houseprice/bin:$PATH"

# 3. Install Airflow with the OFFICIAL constraints file for your Python version.
#    Constraints are mandatory: Airflow pins a large dependency tree.
export PIP_CONSTRAINT=https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt
pip install "apache-airflow==3.3.2"

# 4. Install MLflow and the ML stack under the same constraints
pip install "mlflow==3.16.1" matplotlib pytest

# 5. Confirm
pip check
```

> **Put the virtual environment on the WSL filesystem, not on `/mnt/d`.**
> Installing ~200 packages onto the Windows 9p/DrvFs mount is slow and produced a
> truncated MLflow installation during development (missing `mlflow.sklearn`,
> `mlflow.models` and the `mlflow` CLI). `~/.venvs/houseprice` is on ext4 and is
> fast and reliable.

If you use a different Python version, change `constraints-3.12.txt` to
`constraints-3.10.txt` / `3.11` / `3.13` / `3.14` as appropriate. Airflow 3.3.2
supports Python 3.10 – 3.14.

---

## 9. Start MLflow

```bash
cd "/mnt/d/EA Sports/California House Price Prediction"
bash scripts/start_mlflow.sh
```

Equivalent manual command:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri "sqlite:///$PWD/mlflow/mlflow.db" \
  --default-artifact-root "$PWD/mlflow/artifacts"
```

Stop it with `bash scripts/stop_mlflow.sh`. **Always stop the server before
deleting `mlflow/mlflow.db`** (see [Troubleshooting](#17-troubleshooting)).

---

## 10. Start Airflow

```bash
cd "/mnt/d/EA Sports/California House Price Prediction"
bash scripts/start_airflow.sh
```

The script sets `AIRFLOW_HOME`, points Airflow at this project's `dags/` folder,
runs `airflow db migrate`, and starts `airflow standalone` (API server +
scheduler + DAG processor + triggerer).

Equivalent manual commands:

```bash
export AIRFLOW_HOME="$HOME/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
airflow db migrate
airflow standalone
```

**Get the login password.** Airflow 3 generates a random password for user
`admin` on first start:

```bash
grep "Password for user" logs/airflow_standalone.log
# or
cat "$AIRFLOW_HOME/simple_auth_manager_passwords.json.generated"
```

Stop it with `bash scripts/stop_airflow.sh`.

---

## 11. Access the UIs

| UI | URL | Login |
|---|---|---|
| Airflow | http://localhost:8080 | user `admin`, generated password |
| MLflow | http://localhost:5000 | none required |

WSL2 forwards `localhost` to Windows, so both open directly in your Windows
browser. Verify from PowerShell:

```powershell
Invoke-WebRequest http://localhost:8080/api/v2/monitor/health -UseBasicParsing
Invoke-WebRequest http://localhost:5000/health -UseBasicParsing
```

---

## 12. Run the Pipeline

### Option A — trigger the DAG in the Airflow UI (recommended)

1. Open http://localhost:8080 and log in.
2. Go to **DAGs** and find `house_price_pipeline`.
3. If the toggle shows **paused**, click it to unpause.
4. Open the DAG and click **Trigger DAG** (top right), then **Confirm**.
5. Watch the Grid or Graph view: the five tasks run left to right.
6. Click any task to read its **Logs**.

### Option B — trigger from the command line

```bash
export PATH="$HOME/.venvs/houseprice/bin:$PATH"
export AIRFLOW_HOME="$HOME/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"

airflow dags unpause house_price_pipeline
airflow dags trigger house_price_pipeline
airflow tasks states-for-dag-run house_price_pipeline <run_id>
```

### Option C — run the ML pipeline without Airflow

```bash
bash scripts/run_pipeline.sh              # log to the MLflow server
bash scripts/run_pipeline.sh --no-mlflow  # no tracking
bash scripts/run_pipeline.sh --local-mlflow   # track into ./mlruns
```

The DAG is scheduled `@daily` with `catchup=False` and `max_active_runs=1`, so
manual triggers never overlap.

---

## 13. MLflow Tracking

Experiment name: **`california-house-price-prediction`**

The experiment is created on first use and reused afterwards — **runs are never
overwritten**, so each pipeline execution adds two new runs:

```text
DAG run 1  -> Run 1  Random Forest Regressor
            -> Run 2  Gradient Boosting Regressor
DAG run 2  -> Run 3  Random Forest Regressor
            -> Run 4  Gradient Boosting Regressor
...
```

Each run contains:

| Category | Contents |
|---|---|
| Parameters | `model_type`, `n_estimators`, `max_depth`, `random_state`, `n_features`; plus `learning_rate` for Gradient Boosting |
| Metrics | `mae`, `rmse`, `r2` (full float precision, not rounded) |
| Artifacts | `model/` (MLflow model), `feature_names.json`, `model_summary.json`, `plots/<model>_actual_vs_predicted.png` |

Metrics are logged to the **same run** that holds the model: `train_models`
creates the run and logs parameters + model, then `evaluate_models` reopens that
run id and adds metrics + plot. `train_models` returns the run ids through XCom.

To find the comparison table in the UI: open the experiment, select both runs,
and choose **Compare** — or read `artifacts/reports/model_comparison.csv`,
which the pipeline rewrites on every evaluation.

---

## 14. Metrics

**MAE — Mean Absolute Error.** The average absolute difference between
predicted and actual values, in the same units as the target. In this dataset
1 unit = $100,000, so an MAE of 0.366 means predictions are off by about
**$36,600** on average. Lower is better, and it is easy to interpret.

**RMSE — Root Mean Squared Error.** Squares each error before averaging, so
large mistakes are penalised more heavily. It is also in target units, and it is
always greater than or equal to MAE. Useful for spotting models that are
occasionally very wrong. Lower is better.

**R² — Coefficient of Determination.** The proportion of variance in the target
explained by the model. `1.0` is perfect, `0.0` means no better than always
predicting the mean, and negative values mean worse than the mean. Higher is
better.

```text
RMSE  = sqrt(mean((y_true - y_pred)^2))
R^2   = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
```

No artificial "accuracy" metric is defined here — this is a regression problem.

---

## 15. Results

Produced by an actual Airflow DAG run on this machine
(`test_size=0.2`, `random_state=42`, 20,640 samples, 4,128 test rows):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest Regressor | 0.366316 | 0.544478 | 0.773768 |
| Gradient Boosting Regressor | 0.371643 | 0.542215 | 0.775645 |

Both models land around R² ≈ 0.77, which is the normal range for this dataset
with untuned tree models — roughly 23% of the variance stays unexplained.

How to see the real numbers yourself:

* MLflow UI → experiment `california-house-price-prediction` → open a run.
* `artifacts/reports/model_comparison.csv`
* `artifacts/reports/model_comparison.json`
* `bash scripts/run_pipeline.sh --no-mlflow` prints the table to the console.

Because `random_state=42` is fixed, re-running reproduces the same values
exactly (verified across 5 DAG runs). Different scikit-learn versions may shift
the last digits.

The code deliberately does **not** declare a winner — comparing the numbers is
the point of the exercise. As it happens the two models trade places: Random
Forest has the lower MAE, Gradient Boosting the lower RMSE and marginally higher
R².

---

## 16. Tests

```bash
export PATH="$HOME/.venvs/houseprice/bin:$PATH"
python -m pytest
```

33 tests covering:

* **Dataset** — loads, is not empty (20,640 × 8), expected features exist, no
  missing values, numeric target.
* **Validation** — passes on real data; raises `DataValidationError` for empty
  frames, missing columns, length mismatch and NaNs.
* **Preprocessing** — 80/20 split sizes, train/test index overlap is empty (no
  data leakage), split is reproducible and seed-sensitive.
* **Models** — correct class, `random_state` applied, trains and predicts,
  deterministic across fits, unknown model name raises.
* **Metrics** — MAE/RMSE/R² on known values, perfect predictions give 0 error,
  values match scikit-learn, full precision preserved, shape mismatch and empty
  input raise.
* **Plot** — file is created.
* **MLflow tracking** — `log_run` + `log_evaluation` round-trip against a
  local sqlite tracking URI (pins the exact run dict keys the DAG passes
  between `train_models` and `evaluate_models` via XCom), experiment reuse
  does not create duplicates, empty results are rejected.
* **Airflow** — the DAG file imports, exposes the right `dag_id` and the exact
  five task ids, and the dependency graph is linear
  `load → validate → prepare → train → evaluate`.

The tests do not need a running MLflow *server* — the tracking tests point at
a temporary local sqlite file instead.

---

## 17. Troubleshooting

Problems actually hit while building this project, and their fixes.

**1. `AttributeError: module 'os' has no attribute 'register_at_fork'`**
Airflow 3 cannot start on Windows at all. Run everything in WSL2 (see
[Installation](#8-installation)).

**2. `python3 -m venv` → "ensurepip is not available / you may need to use sudo"**
WSL's `python3-venv` package is missing and `sudo` needs a password. Create the
venv with `--without-pip` and bootstrap pip with `get-pip.py` — no root needed.

**3. `No module named 'mlflow.sklearn'` and no `mlflow` command on the PATH**
The MLflow install was truncated because the venv lived on `/mnt/d` (Windows
9p/DrvFs). Recreate the venv inside WSL (`~/.venvs/houseprice`) and reinstall.
Check with `pip check` and `python -c "import mlflow.sklearn"`.

**4. `The saved sklearn model references untrusted types ... sklearn.tree._tree.Tree`**
MLflow ≥ 3 serialises scikit-learn models with `skops`, which refuses to
deserialise the tree node-storage type unless it is declared trusted. Since this
pipeline trains the models itself, that type is safe to trust:

```python
mlflow.sklearn.log_model(
    sk_model=model, name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
```

**5. `'RunInfo' object has no attribute 'run_uri'`**
Removed in MLflow 3. Build the link yourself:
`f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"`.

**6. MLflow tasks fail with `too many 503 error responses`**
`mlflow/mlflow.db` was deleted while the server still had it open. Run
`bash scripts/stop_mlflow.sh`, delete `mlflow/`, then
`bash scripts/start_mlflow.sh`.

**7. `artifacts/reports` and runs are visible in one place but the UI is empty**
The Airflow tasks and the MLflow UI are using different tracking URIs. Check that
`MLFLOW_TRACKING_URI` is `http://127.0.0.1:5000` for both. The DAG also sets this
value as a default, so it works even if the variable is missing.

**8. `curl ... 8080` succeeds after "stopping" Airflow**
`airflow standalone` forks child processes (api-server, scheduler, DAG processor,
triggerer, workers). Killing only the parent leaves them holding port 8080 —
`scripts/stop_airflow.sh` matches and kills every component.

**9. `Could not import graphviz` warning**
Harmless. It only means the DAG graph view falls back to a non-graphical
rendering. Install `graphviz` + system graphviz only if you want the pretty
diagram.

**10. `KeyError: 'dag_run_id'` when triggering via the Airflow REST API**
Airflow 3 requires `logical_date` in the POST body. Use the CLI
(`airflow dags trigger`) or send `{"dag_run_id": "...", "logical_date": null}`.

**11. `Database migration required` / `no such table: dag_bundle`**
Run `airflow db migrate` (or use `scripts/start_airflow.sh`, which does it for
you).

**12. DAG does not appear in the UI**
Confirm `AIRFLOW__CORE__DAGS_FOLDER` points at this project's `dags/` folder and
check `airflow dags list-import-errors`.

**13. The first `fetch_california_housing()` needs internet**
It downloads once, then caches to `data/raw/california_housing.pkl`. Later runs
and the tests are offline.

**14. A run in the MLflow UI has params + model but never got metrics**
`train_models` has `retries: 1`. If it fails partway through the model loop (an
MLflow error on the second model, say), the whole task retries and re-trains +
re-logs both models under new run ids; only the final attempt's run ids reach
`evaluate_models` via XCom. The earlier attempt's run(s) are left in MLflow
with params/model but no metrics — harmless, but confusing when debugging.

**15. `stop_mlflow.sh` / `stop_airflow.sh` only target this project**
Both scripts match process patterns scoped to this project's paths (the
`mlflow.db` path for MLflow; the standard `airflow standalone` component names
for Airflow, which share one metadata DB machine-wide by design). If you run a
second MLflow project on a different port at the same time, stopping one does
not touch the other.

---

## 18. Future Improvements

Ideas this project deliberately leaves out, in rough order of value:

* **MLflow Model Registry** — register runs and promote the chosen model to
  `staging` / `production` with an alias.
* **Automated model selection** — a DAG task that picks the best run by RMSE
  (using the MLflow client) instead of leaving the choice to a human.
* **Hyperparameter tuning** — `RandomizedSearchCV` inside the pipeline, logging
  each trial as its own run.
* **Containerisation** — Dockerfiles for Airflow and MLflow plus
  `docker compose up` so setup becomes one command.
* **Cloud deployment** — managed Airflow (MWAA/Composer) and a hosted MLflow, or
  switching the backend store to PostgreSQL and artifacts to S3.
* **CI/CD** — run `pytest` on every pull request; add a job that triggers the
  DAG on `main` and publishes the resulting metrics.
* **Model monitoring** — schedule a scoring job on new data and alert when MAE
  drifts beyond a threshold.
* **Data versioning** — attach dataset checksums/versions to each run via
  MLflow data logging or DVC.
* **Database-backed metadata** — replace the local SQLite store with PostgreSQL
  so several Airflow workers can share state.
* **Serving** — a small prediction service loading the model straight from
  MLflow with `runs:/<run_id>/model` or `models:/<name>@<version>`.

---

## License

Provided for learning purposes.
