import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


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

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "train_33class.csv"
)

SVM_C = 0.5
RANDOM_STATE = 42


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["merged_category"]

print("Training data loaded")
print("Training rows:", len(train_df))
print("Classes:", y_train.nunique())


# ============================================================
# 2. TF-IDF FEATURE ENGINEERING
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

print("TF-IDF training shape:",
      X_train_tfidf.shape)


# ============================================================
# 3. TRAIN LINEAR SVM
# ============================================================

model = LinearSVC(
    C=SVM_C,
    random_state=RANDOM_STATE,
    max_iter=5000
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Linear SVM training completed")
print("C:", SVM_C)


# ============================================================
# 4. SAVE MODEL ARTIFACTS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "news_svm_33class.joblib"
)

vectorizer_path = os.path.join(
    MODEL_DIR,
    "news_tfidf_33class.joblib"
)

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nModel artifacts saved:")
print(model_path)
print(vectorizer_path)
