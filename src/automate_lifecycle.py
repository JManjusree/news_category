import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL = "NewsCategory_33Class_SVM"
CANDIDATE = Path("reports/latest_candidate.json")
LIFECYCLE = Path("reports/model_lifecycle.json")

def main():
    if not CANDIDATE.exists():
        raise FileNotFoundError(
            "reports/latest_candidate.json not found. Run train_registry.py first."
        )

    with open(CANDIDATE) as f:
        candidate = json.load(f)

    client = MlflowClient()

    # This lab uses Macro-F1 as the documented promotion metric.
    candidate_score = float(candidate["macro_f1"])

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL}'"
    )
    if not versions:
        raise RuntimeError("No registered model versions were found.")

    current_version = max(versions, key=lambda v: int(v.version))

    # Determine the best recorded registered version by its run metric.
    best_version = current_version
    best_score = candidate_score

    for v in versions:
        try:
            run = client.get_run(v.run_id)
            score = run.data.metrics.get("test_macro_f1")
            if score is not None and float(score) > best_score:
                best_score = float(score)
                best_version = v
        except Exception:
            continue

    # Use aliases instead of deprecated MLflow stages.
    client.set_registered_model_alias(
        REGISTERED_MODEL, "Champion", best_version.version
    )
    client.set_registered_model_alias(
        REGISTERED_MODEL, "Candidate", current_version.version
    )

    lifecycle = {
        "model_name": REGISTERED_MODEL,
        "champion_version": int(best_version.version),
        "champion_macro_f1": float(best_score),
        "candidate_version": int(current_version.version),
        "candidate_macro_f1": candidate_score,
        "promotion_metric": "test_macro_f1",
        "aliases": {
            "Champion": int(best_version.version),
            "Candidate": int(current_version.version)
        }
    }

    LIFECYCLE.parent.mkdir(exist_ok=True)
    with open(LIFECYCLE, "w") as f:
        json.dump(lifecycle, f, indent=2)

    print("[SUCCESS] Model lifecycle updated.")
    print(f"[INFO] Champion version: {best_version.version}")
    print(f"[INFO] Champion Macro-F1: {best_score:.4f}")

if __name__ == "__main__":
    main()
