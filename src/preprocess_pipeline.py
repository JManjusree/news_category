
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(
    BASE_DIR, "data", "raw", "News_Category_Dataset_v3.json"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR, "data", "processed"
)

RANDOM_STATE = 42
TEST_SIZE = 0.15
EXPECTED_CLASSES = 33


# ============================================================
# 2. CATEGORY MAPPING
# ============================================================

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

    # Keep WEDDINGS as a separate category
    "WEDDINGS": "WEDDINGS",
    "DIVORCE": "RELATIONSHIPS",
}


# ============================================================
# 3. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# 4. MAIN PREPROCESSING FUNCTION
# ============================================================

def main():

    print("=" * 65)
    print("[INFO] Starting production preprocessing...")
    print("=" * 65)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Check raw dataset
    # --------------------------------------------------------

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}"
        )

    df = pd.read_json(RAW_PATH, lines=True)

    required_columns = [
        "headline",
        "short_description",
        "authors",
        "date",
        "category"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"[INFO] Original dataset rows: {len(df)}")

    # --------------------------------------------------------
    # Step 2: Handle missing values
    # --------------------------------------------------------

    df["headline"] = (
        df["headline"].fillna("").astype(str)
    )

    df["short_description"] = (
        df["short_description"].fillna("").astype(str)
    )

    df["authors"] = (
        df["authors"].fillna("unknown").astype(str)
    )

    df["authors"] = df["authors"].replace("", "unknown")

    # Remove rows with missing category labels
    df = df[df["category"].notna()].copy()

    df["category"] = df["category"].astype(str).str.strip()

    # --------------------------------------------------------
    # Step 3: Process date and year
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year"] = (
        df["date"].dt.year.fillna(0).astype(int)
    )

    # --------------------------------------------------------
    # Step 4: Create cleaned text
    # --------------------------------------------------------

    df["text"] = clean_text(
        df["headline"] + " " + df["short_description"]
    )

    # --------------------------------------------------------
    # Step 5: Create merged categories
    # IMPORTANT: This must happen before filtering.
    # --------------------------------------------------------

    df["merged_category"] = (
        df["category"]
        .map(MERGE_MAP)
        .fillna(df["category"])
    )

    # --------------------------------------------------------
    # Step 6: Remove records with empty text or labels
    # --------------------------------------------------------

    rows_before_filter = len(df)

    df = df[
        df["text"].notna()
        & df["text"].str.strip().ne("")
        & df["merged_category"].notna()
        & df["merged_category"].astype(str).str.strip().ne("")
    ].copy()

    rows_removed = rows_before_filter - len(df)

    print(
        f"[INFO] Rows removed due to empty text/labels: "
        f"{rows_removed}"
    )

    print(f"[INFO] Rows after cleaning: {len(df)}")

    if df.empty:
        raise ValueError(
            "No valid records remain after text cleaning."
        )

    # --------------------------------------------------------
    # Step 7: Validate 33 merged categories
    # --------------------------------------------------------

    classes = df["merged_category"].nunique()

    print(f"[INFO] Merged category count: {classes}")

    if classes != EXPECTED_CLASSES:
        print("\n[ERROR] Unexpected category count.")
        print("\nAvailable categories:")

        for category in sorted(df["merged_category"].unique()):
            print(category)

        raise ValueError(
            f"Expected {EXPECTED_CLASSES} merged classes, "
            f"but found {classes}."
        )

    print("[INFO] All 33 categories validated successfully.")

    # --------------------------------------------------------
    # Step 8: Stratified train/test split
    # --------------------------------------------------------

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["merged_category"]
    )

    columns = [
        "text",
        "authors",
        "year",
        "merged_category"
    ]

    train_df = df.loc[train_idx, columns].copy()
    test_df = df.loc[test_idx, columns].copy()

    # --------------------------------------------------------
    # Step 9: Validate train/test datasets
    # --------------------------------------------------------

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test dataset is empty."
        )

    if train_df["text"].isna().any():
        raise ValueError(
            "Training dataset contains null text."
        )

    if test_df["text"].isna().any():
        raise ValueError(
            "Test dataset contains null text."
        )

    train_classes = set(train_df["merged_category"].unique())
    test_classes = set(test_df["merged_category"].unique())

    if len(train_classes) != EXPECTED_CLASSES:
        raise ValueError(
            "Training dataset does not contain all 33 categories."
        )

    if train_classes != test_classes:
        raise ValueError(
            "Train and test category sets do not match."
        )

    # --------------------------------------------------------
    # Step 10: Save processed datasets
    # --------------------------------------------------------

    train_path = os.path.join(
        PROCESSED_DIR,
        "train_33class.csv"
    )

    test_path = os.path.join(
        PROCESSED_DIR,
        "test_33class.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # --------------------------------------------------------
    # Step 11: Generate metadata
    # --------------------------------------------------------

    metadata = {
        "project": "News Category Classification",
        "lab": "Lab 7 - ML Pipeline Engineering",
        "total_rows_before_cleaning": int(rows_before_filter),
        "rows_removed": int(rows_removed),
        "total_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "original_classes": int(df["category"].nunique()),
        "merged_classes": int(classes),
        "expected_classes": EXPECTED_CLASSES,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target": "merged_category",
        "text_source": "headline + short_description",
        "category_mapping": MERGE_MAP,
        "train_file": "train_33class.csv",
        "test_file": "test_33class.csv"
    }

    metadata_path = os.path.join(
        PROCESSED_DIR,
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    # --------------------------------------------------------
    # Step 12: Display final results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("[SUCCESS] Production preprocessing completed.")
    print("=" * 65)

    print(f"[INFO] Total rows : {len(df)}")
    print(f"[INFO] Train rows : {len(train_df)}")
    print(f"[INFO] Test rows  : {len(test_df)}")
    print(f"[INFO] Classes    : {classes}")
    print(f"[INFO] Test ratio : {len(test_df) / len(df):.2%}")

    print("\n[INFO] Generated files:")
    print(f"[INFO] {train_path}")
    print(f"[INFO] {test_path}")
    print(f"[INFO] {metadata_path}")

    print("=" * 65)


# ============================================================
# 5. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
