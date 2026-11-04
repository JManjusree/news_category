"""Configuration for the News Category Lab 7 pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data/raw/News_Category_Dataset_v3.json"
PROCESSED = PROJECT_ROOT / "data/processed"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
ARTIFACTS = PROJECT_ROOT / "artifacts/lab7"
LOGS = PROJECT_ROOT / "logs"

TRAIN_CSV = PROCESSED / "train_33class.csv"
TEST_CSV = PROCESSED / "test_33class.csv"
MODEL_PATH = MODELS / "lab7_news_pipeline.joblib"
METRICS_PATH = REPORTS / "lab7_evaluation_report.json"
INFERENCE_PATH = ARTIFACTS / "inference_predictions.csv"
RUN_REPORT_PATH = REPORTS / "lab7_pipeline_report.json"
RUN_LOG_PATH = LOGS / "news_category_lab7_pipeline_run_log.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15
EXPECTED_CLASSES = 33

# Lab 8: GitHub Actions CI trigger verification.