import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# MLOPS CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

EVALUATION_DIR = os.path.join(
    BASE_DIR, "reports"
)

os.makedirs(EVALUATION_DIR, exist_ok=True)


TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "test_33class.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "news_svm_33class.joblib"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "news_tfidf_33class.joblib"
)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_PATH)

X_test = test_df["text"].fillna("")
y_test = test_df["merged_category"]


# ============================================================
# 2. LOAD TRAINED ARTIFACTS
# ============================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ============================================================
# 3. TRANSFORM TEST DATA
# ============================================================

X_test_tfidf = vectorizer.transform(X_test)

print("Test TF-IDF shape:",
      X_test_tfidf.shape)


# ============================================================
# 4. PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 5. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEXT ONLY + MERGED LABELS")
print("=" * 70)

print("Classes:", y_test.nunique())
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1:.3f}")
print(f"Weighted-F1: {weighted_f1:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 7. SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{
    "Variant": "Text only, merged labels",
    "Classes": y_test.nunique(),
    "Accuracy": accuracy,
    "Macro-F1": macro_f1,
    "Weighted-F1": weighted_f1,
    "Test_Rows": len(y_test)
}])

metrics_path = os.path.join(
    EVALUATION_DIR,
    "news_33class_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 8. SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    EVALUATION_DIR,
    "classification_report_33class.csv"
)

report_df.to_csv(
    report_path
)

print("\nEvaluation artifacts saved:")
print(metrics_path)
print(report_path)
