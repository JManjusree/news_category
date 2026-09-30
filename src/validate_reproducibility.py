
import hashlib
import csv
import json
import os
import subprocess
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(
    REPORT_DIR,
    "news_33class_metrics.csv"
)


def file_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("[INFO] Re-running preprocessing...")
    subprocess.run(
        [
            sys.executable,
            os.path.join(BASE_DIR, "src", "preprocess.py")
        ],
        check=True,
    )

    print("[INFO] Re-running training...")
    subprocess.run(
        [
            sys.executable,
            os.path.join(BASE_DIR, "src", "train_mlflow.py")
        ],
        check=True,
    )

    print("[INFO] Re-running evaluation...")
    subprocess.run(
        [
            sys.executable,
            os.path.join(BASE_DIR, "src", "evaluate.py")
        ],
        check=True,
    )

    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Evaluation metrics were not generated: {METRICS_PATH}"
        )

    print("[INFO] Reading evaluation metrics...")

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(
            "Evaluation metrics CSV is empty."
        )

    print(
        f"[INFO] CSV columns: {list(rows[0].keys())}"
    )

    # evaluate.py creates:
    # Variant, Classes, Accuracy, Macro-F1, Weighted-F1, Test_Rows

    row = rows[0]

    try:
        accuracy = float(row["Accuracy"])
        macro_f1 = float(row["Macro-F1"])
    except KeyError as e:
        raise KeyError(
            f"Required metric column not found: {e}"
        )

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "news_svm_33class.joblib"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file was not generated: {model_path}"
        )

    result = {
        "random_state": 42,
        "test_size": 0.15,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "model_sha256": file_hash(model_path),
        "status": "PASS",
    }

    reproducibility_path = os.path.join(
        REPORT_DIR,
        "reproducibility.json"
    )

    with open(
        reproducibility_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=4
        )

    print()
    print("[SUCCESS] Reproducibility validation completed.")
    print(f"[INFO] Accuracy: {accuracy:.4f}")
    print(f"[INFO] Macro-F1: {macro_f1:.4f}")
    print(
        f"[INFO] Model SHA256: "
        f"{result['model_sha256']}"
    )
    print("[INFO] Fixed random_state = 42")
    print("[INFO] Fixed test_size = 0.15")
    print(
        f"[INFO] Saved: {reproducibility_path}"
    )


if __name__ == "__main__":
    main()

