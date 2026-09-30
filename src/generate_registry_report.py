import json
from pathlib import Path
from datetime import datetime, timezone

CANDIDATE = Path("reports/latest_candidate.json")
LIFECYCLE = Path("reports/model_lifecycle.json")
OUTPUT = Path("reports/registry_report.json")

def main():
    if not CANDIDATE.exists() or not LIFECYCLE.exists():
        raise FileNotFoundError(
            "Registry inputs missing. Run train_registry.py and automate_lifecycle.py first."
        )

    with open(CANDIDATE) as f:
        candidate = json.load(f)
    with open(LIFECYCLE) as f:
        lifecycle = json.load(f)

    report = {
        "project": "News Category Classification",
        "task": "33-class text classification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registered_model": lifecycle["model_name"],
        "champion": {
            "version": lifecycle["champion_version"],
            "macro_f1": lifecycle["champion_macro_f1"],
            "alias": "Champion"
        },
        "candidate": {
            "version": lifecycle["candidate_version"],
            "macro_f1": lifecycle["candidate_macro_f1"],
            "accuracy": candidate["accuracy"],
            "alias": "Candidate"
        },
        "reproducibility": {
            "random_state": candidate["random_state"],
            "test_size": candidate["test_size"],
            "num_classes": candidate["num_classes"]
        },
        "deployment_reference": {
            "mlflow_model_uri": (
                f"models:/{lifecycle['model_name']}/"
                f"{lifecycle['champion_version']}"
            ),
            "champion_alias_uri": (
                f"models:/{lifecycle['model_name']}@Champion"
            )
        }
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Registry deployment report written to {OUTPUT}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
