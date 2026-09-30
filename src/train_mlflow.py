import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train_33class.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

RANDOM_STATE = 42
SVM_C = 0.5
MAX_FEATURES = 50000


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df["text"].fillna("")
    y_train = train_df["merged_category"]

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LinearSVC(
        C=SVM_C,
        random_state=RANDOM_STATE,
        max_iter=5000,
    )

    experiment_name = "News_Category_33Class"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="tfidf_linear_svc_33class"):
        model.fit(X_train_tfidf, y_train)

        train_pred = model.predict(X_train_tfidf)

        train_accuracy = accuracy_score(y_train, train_pred)
        train_macro_f1 = f1_score(
            y_train, train_pred, average="macro", zero_division=0
        )

        mlflow.log_param("model", "LinearSVC")
        mlflow.log_param("feature_extractor", "TF-IDF")
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("stop_words", "english")
        mlflow.log_param("sublinear_tf", True)
        mlflow.log_param("C", SVM_C)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("classes", 33)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_macro_f1", train_macro_f1)

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="NewsCategory_33Class_SVM",
        )

        joblib.dump(
            model,
            os.path.join(MODEL_DIR, "news_svm_33class.joblib"),
        )
        joblib.dump(
            vectorizer,
            os.path.join(MODEL_DIR, "news_tfidf_33class.joblib"),
        )

        with open(
            os.path.join(REPORT_DIR, "train_metrics.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "train_accuracy": train_accuracy,
                    "train_macro_f1": train_macro_f1,
                },
                f,
                indent=4,
            )

        print("[SUCCESS] MLflow training completed.")
        print(f"[INFO] Train Accuracy: {train_accuracy:.4f}")
        print(f"[INFO] Train Macro-F1: {train_macro_f1:.4f}")


if __name__ == "__main__":
    main()
