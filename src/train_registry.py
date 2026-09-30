import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

DATA = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TRAIN = DATA / "train_33class.csv"
TEST = DATA / "test_33class.csv"

EXPERIMENT = "News_Category_33Class_Registry"
REGISTERED_MODEL = "NewsCategory_33Class_SVM"

def main():
    if not TRAIN.exists() or not TEST.exists():
        raise FileNotFoundError(
            "Processed data not found. Run the Lab-5 News pipeline first."
        )

    import pandas as pd
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)

    required = {"text", "merged_category"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"Expected columns {required} in train/test CSV files.")

    mlflow.set_experiment(EXPERIMENT)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(C=0.5, random_state=42, max_iter=5000))
    ])

    with mlflow.start_run(run_name="news_33class_registry_candidate") as run:
        pipeline.fit(train["text"].fillna(""), train["merged_category"])

        pred = pipeline.predict(test["text"].fillna(""))
        acc = accuracy_score(test["merged_category"], pred)
        macro_f1 = f1_score(test["merged_category"], pred, average="macro")

        mlflow.log_param("task", "News Category 33-class classification")
        mlflow.log_param("text_features", "headline + short_description")
        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("C", 0.5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.15)
        mlflow.log_param("num_classes", 33)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_macro_f1", macro_f1)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL
        )

        model_path = MODEL_DIR / f"news_33class_{run.info.run_id}.joblib"
        joblib.dump(pipeline, model_path)

        candidate = {
            "run_id": run.info.run_id,
            "model_name": REGISTERED_MODEL,
            "model_path": str(model_path),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "random_state": 42,
            "test_size": 0.15,
            "num_classes": 33
        }
        Path("reports").mkdir(exist_ok=True)
        with open("reports/latest_candidate.json", "w") as f:
            json.dump(candidate, f, indent=2)

        print(f"[SUCCESS] Registered candidate model: {REGISTERED_MODEL}")
        print(f"[INFO] Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")
        print(f"[INFO] Run ID={run.info.run_id}")

if __name__ == "__main__":
    main()
