"""Sequential Lab 7 workflow, following the uploaded churn Lab 7 runner pattern."""
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from lab7_config import LOGS, RUN_LOG_PATH
from lab7_components import (
    setup_logging, validate_raw_data, validate_processed_data, train_model,
    evaluate_model, run_inference_smoke_test, run_reproducibility_check,
    write_pipeline_report,
)

def run_existing_preprocessing():
    """Execute the project's established 33-class preprocessing stage."""
    script = PROJECT_ROOT / "src" / "preprocess_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(f"Existing preprocessing script not found: {script}")
    completed = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Preprocessing stage exited with code {completed.returncode}")
    return {"script": str(script), "exit_code": completed.returncode}

def main():
    started = time.time()
    setup_logging(RUN_LOG_PATH)
    logging.info("=" * 64)
    logging.info("Starting News Category Lab 7 ML Pipeline")
    stages = [
        ("01 Raw data validation", validate_raw_data),
        ("02 Preprocessing", run_existing_preprocessing),
        ("03 Processed data validation", validate_processed_data),
        ("04 Model training", train_model),
        ("05 Model evaluation", evaluate_model),
        ("06 Inference smoke test", run_inference_smoke_test),
        ("07 Reproducibility check", run_reproducibility_check),
    ]
    results = []
    status = "PASSED"
    for name, action in stages:
        logging.info("RUNNING: %s", name)
        try:
            output = action()
            results.append({"stage": name, "status": "PASSED", "output": output})
            logging.info("PASSED: %s", name)
        except Exception as exc:
            results.append({"stage": name, "status": "FAILED", "error": str(exc)})
            logging.exception("FAILED: %s", name)
            status = "FAILED"
            break
    report = write_pipeline_report(results, started, status)
    logging.info("Pipeline status: %s", status)
    logging.info("Pipeline report: %s", report["project"])
    return status == "PASSED"

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
