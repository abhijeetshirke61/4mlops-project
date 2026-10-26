import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_DIR = Path(os.environ.get(
    "MLOPS_PROJECT_DIR", str(Path(__file__).resolve().parents[2])
))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocess import download_data, preprocess
from train import train

DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_DIR / "models"
DEPLOYED_DIR = PROJECT_DIR / "deployed"
BEST_METRICS = DEPLOYED_DIR / "best_metrics.json"


def download_task():
    download_data()


def preprocess_task():
    preprocess(DATA_DIR / "data.csv", PROCESSED_DIR)


def train_task():
    train(PROCESSED_DIR, MODELS_DIR)


def evaluate_task():
    with open(MODELS_DIR / "metrics.json", encoding="utf-8") as file:
        new_accuracy = json.load(file)["accuracy"]
    old_accuracy = -1.0
    if BEST_METRICS.exists():
        with open(BEST_METRICS, encoding="utf-8") as file:
            old_accuracy = json.load(file)["accuracy"]
    print(f"New accuracy={new_accuracy:.4f}; previous accuracy={old_accuracy:.4f}")
    return "deploy_model" if new_accuracy > old_accuracy else "skip_deployment"


def deploy_task():
    DEPLOYED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODELS_DIR / "model.joblib", DEPLOYED_DIR / "model.joblib")
    shutil.copy2(MODELS_DIR / "metrics.json", BEST_METRICS)
    print("Model deployed to deployed/")


with DAG(
    dag_id="mlops_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "assignment4"],
) as dag:
    t1 = PythonOperator(task_id="download_data", python_callable=download_task)
    t2 = PythonOperator(task_id="preprocess", python_callable=preprocess_task)
    t3 = PythonOperator(task_id="train_model", python_callable=train_task)
    t4 = BranchPythonOperator(task_id="evaluate_model", python_callable=evaluate_task)
    deploy = PythonOperator(task_id="deploy_model", python_callable=deploy_task)
    skip = EmptyOperator(task_id="skip_deployment")
    t1 >> t2 >> t3 >> t4 >> [deploy, skip]
