import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "News_Category_Dataset_v3.json")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RANDOM_STATE = 42
TEST_SIZE = 0.15

MERGE_MAP = {
    "ARTS": "ARTS & CULTURE",
    "ARTS & CULTURE": "ARTS & CULTURE",
    "CULTURE & ARTS": "ARTS & CULTURE",
    "PARENTING": "PARENTING",
    "PARENTS": "PARENTING",
    "STYLE": "STYLE & BEAUTY",
    "STYLE & BEAUTY": "STYLE & BEAUTY",
    "THE WORLDPOST": "WORLDPOST",
    "WORLDPOST": "WORLDPOST",
    "GREEN": "ENVIRONMENT & GREEN",
    "ENVIRONMENT": "ENVIRONMENT & GREEN",
    "HEALTHY LIVING": "WELLNESS",
    "WELLNESS": "WELLNESS",
    "FOOD & DRINK": "FOOD & DRINK",
    "TASTE": "FOOD & DRINK",
    "COLLEGE": "EDUCATION",
    "EDUCATION": "EDUCATION",
    "WEDDINGS": "RELATIONSHIPS",
    "DIVORCE": "RELATIONSHIPS",
}


def clean_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    print("[INFO] Starting production preprocessing...")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_json(RAW_PATH, lines=True)

    df["headline"] = df["headline"].fillna("").astype(str)
    df["short_description"] = df["short_description"].fillna("").astype(str)
    df["authors"] = df["authors"].fillna("unknown").astype(str)
    df["authors"] = df["authors"].replace("", "unknown")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year.fillna(0).astype(int)

    df["text"] = clean_text(
        df["headline"] + " " + df["short_description"]
    )

    df["merged_category"] = (
        df["category"].map(MERGE_MAP).fillna(df["category"])
    )

    classes = df["merged_category"].nunique()
    if classes != 33:
        raise ValueError(
            f"Expected 33 merged classes, found {classes}. "
            "Check the label mapping."
        )

    # One fixed 15% stratified holdout for reproducible experiments.
    train_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["merged_category"]
    )

    columns = ["text", "authors", "year", "merged_category"]

    train_df = df.iloc[train_idx][columns].copy()
    test_df = df.iloc[test_idx][columns].copy()

    train_path = os.path.join(PROCESSED_DIR, "train_33class.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_33class.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "original_classes": int(df["category"].nunique()),
        "merged_classes": int(classes),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target": "merged_category",
        "text_source": "headline + short_description"
    }

    with open(
        os.path.join(PROCESSED_DIR, "metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("[INFO] Production preprocessing completed.")
    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Test rows : {len(test_df)}")
    print("[INFO] Classes   : 33")


if __name__ == "__main__":
    main()
