"""Reusable Lab 7 components for the 33-class News Category project."""
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lab7_config import (
    ARTIFACTS, EXPECTED_CLASSES, INFERENCE_PATH, METRICS_PATH, MODEL_PATH,
    PROCESSED, RANDOM_STATE, RAW_DATA, REPORTS, RUN_REPORT_PATH, TEST_CSV,
    TEST_SIZE, TRAIN_CSV,
)

def setup_logging(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def validate_raw_data():
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA}")
    frame = pd.read_json(RAW_DATA, lines=True)
    required = {"headline", "short_description", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {sorted(missing)}")
    if frame.empty or frame["category"].isna().any():
        raise ValueError("Raw dataset is empty or contains null categories.")
    return {"rows": int(len(frame)), "columns": list(frame.columns),
            "raw_category_count": int(frame["category"].nunique())}

def validate_processed_data():
    for path in (TRAIN_CSV, TEST_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Expected preprocessing output missing: {path}")
    train, test = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    required = {"text", "merged_category"}
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split missing columns: {sorted(missing)}")
        if frame.empty or frame["text"].isna().any() or frame["merged_category"].isna().any():
            raise ValueError(f"{name} split is empty or has null text/labels.")
    train_classes = set(train["merged_category"].unique())
    test_classes = set(test["merged_category"].unique())
    if len(train_classes) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} train classes; found {len(train_classes)}.")
    if train_classes != test_classes:
        raise ValueError("Train/test category sets do not match.")
    ratio = len(test) / (len(train) + len(test))
    if abs(ratio - TEST_SIZE) > 0.015:
        raise ValueError(f"Expected approximately {TEST_SIZE:.0%} test split; found {ratio:.2%}.")
    return {"train_rows": len(train), "test_rows": len(test),
            "classes": len(train_classes), "test_ratio": ratio}

def build_sklearn_pipeline():
    # The vectorizer and classifier are packaged together for consistent inference.
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  stop_words="english", sublinear_tf=True)),
        ("classifier", LinearSVC(C=0.5, random_state=RANDOM_STATE, max_iter=5000)),
    ])

def train_model():
    train = pd.read_csv(TRAIN_CSV)
    model = build_sklearn_pipeline()
    model.fit(train["text"].astype(str), train["merged_category"].astype(str))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {"model_path": str(MODEL_PATH), "train_rows": len(train),
            "model_sha256": sha256_file(MODEL_PATH)}

def evaluate_model():
    test = pd.read_csv(TEST_CSV)
    model = joblib.load(MODEL_PATH)
    actual = test["merged_category"].astype(str)
    predicted = model.predict(test["text"].astype(str))
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "Lab 7 model evaluation",
        "evaluated_on": "held-out test split",
        "test_rows": int(len(test)),
        "metrics": metrics,
        "classification_report": classification_report(actual, predicted, output_dict=True, zero_division=0),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return metrics

def run_inference_smoke_test():
    test = pd.read_csv(TEST_CSV)
    sample = test.head(5)
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(sample["text"].astype(str))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": sample["text"], "actual_category": sample["merged_category"],
                  "predicted_category": predictions}).to_csv(INFERENCE_PATH, index=False)
    return {"sample_rows": len(sample), "predictions_saved_to": str(INFERENCE_PATH)}

def run_reproducibility_check():
    """Retrain twice and compare predictions + serialized model hashes."""
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    hashes, prediction_sets = [], []
    for _ in range(2):
        model = build_sklearn_pipeline()
        model.fit(train["text"].astype(str), train["merged_category"].astype(str))
        hashes.append(hashlib.sha256(joblib.hash(model).encode("utf-8")).hexdigest())
        prediction_sets.append(model.predict(test["text"].astype(str)).tolist())
    predictions_equal = prediction_sets[0] == prediction_sets[1]
    hashes_equal = hashes[0] == hashes[1]
    result = {"repetitions": 2, "predictions_identical": predictions_equal,
              "model_state_hashes_identical": hashes_equal,
              "status": "PASSED" if predictions_equal and hashes_equal else "FAILED"}
    if not predictions_equal:
        raise RuntimeError("Repeated training produced different predictions.")
    return result

def write_pipeline_report(stage_results, started_at, status):
    report = {
        "project": "News Category Classification",
        "lab": "Lab 7 - ML Pipeline Engineering",
        "overall_status": status,
        "duration_seconds": round(time.time() - started_at, 3),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "stages": stage_results,
        "artifacts": {
            "model": str(MODEL_PATH),
            "evaluation": str(METRICS_PATH),
            "inference": str(INFERENCE_PATH),
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUN_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
