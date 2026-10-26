# Assignment 4 — MLOps: CI/CD Pipeline and Airflow Workflow

## Objective
Automate an Iris machine-learning workflow using Python, GitHub Actions, Docker, and Apache Airflow.

## Project structure
- `src/preprocess.py`: creates the Iris CSV and train/test files.
- `src/train.py`: trains a Random Forest classifier and saves model/metrics.
- `src/predict.py`: loads the model and predicts an Iris class.
- `tests/test_model.py`: tests the pipeline.
- `.github/workflows/mlops.yml`: CI, training, and tag-based container workflow.
- `airflow/dags/mlops_dag.py`: Airflow DAG with conditional deployment.

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
pytest -v
python src/predict.py
```

## GitHub Actions
Create a GitHub repository and push this project to the `main` branch. Pull requests run lint/tests; pushes to `main` train the model and upload an artifact. A version tag such as `v1.0.0` builds and pushes a container to GitHub Container Registry and runs it as a staging smoke test.

## Airflow
Airflow is easiest to run in a separate Docker Compose environment. Copy or mount this repository into the Airflow container and ensure `pandas`, `scikit-learn`, and `joblib` are installed there. Set `MLOPS_PROJECT_DIR` to the project directory if needed. The DAG is manually triggered from the Airflow UI.

## Important
The workflow's staging step runs the container as a smoke test; it is not a public production deployment. Replace it with a real staging service if your instructor requires one.

CI pipeline test.
