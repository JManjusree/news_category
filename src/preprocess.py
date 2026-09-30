import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# MLOPS CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(
    BASE_DIR, "data", "raw", "News_Category_Dataset_v3.json"
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.15


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_json(RAW_PATH, lines=True)

print("Raw data loaded successfully")
print("Rows:", len(df))
print("Original classes:", df["category"].nunique())


# ============================================================
# 2. DATA VALIDATION
# ============================================================

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


# ============================================================
# 3. DATA CLEANING
# ============================================================

for col in ["headline", "short_description", "authors"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["headline"] = df["headline"].replace(
    "", "unknown"
)

df["short_description"] = df["short_description"].replace(
    "", "no description"
)

df["authors"] = df["authors"].replace(
    "", "unknown"
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df["year"] = (
    df["date"]
    .dt.year
    .fillna(0)
    .astype(int)
)


# ============================================================
# 4. CREATE TEXT FEATURE
# ============================================================

df["text"] = (
    df["headline"] + " " +
    df["short_description"]
).str.lower()

df["text"] = (
    df["text"]
    .str.replace(r"[^a-z\s]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("\nOriginal category list:")
print(sorted(df["category"].unique()))

print(
    "\nOriginal class count:",
    df["category"].nunique()
)


# ============================================================
# 5. MERGE 42 LABELS -> 33 LABELS
# ============================================================

# IMPORTANT:
# We intentionally DO NOT merge:
# ARTS
# ARTS & CULTURE
# CULTURE & ARTS
#
# This prevents the previous 42 -> 32 reduction.
# The following 9 merges give the required 33 classes.

# ============================================================
# 5. MERGE 42 LABELS -> 33 LABELS
# ============================================================

merge_map = {

    # 1. Arts
    "ARTS": "ARTS & CULTURE",

    # 2. Parenting
    "PARENTS": "PARENTING",

    # 3. Style
    "STYLE": "STYLE & BEAUTY",

    # 4. WorldPost
    "THE WORLDPOST": "WORLDPOST",

    # 5. Environment
    "GREEN": "ENVIRONMENT",

    # 6. Wellness
    "HEALTHY LIVING": "WELLNESS",

    # 7. Food
    "TASTE": "FOOD & DRINK",

    # 8. Education
    "COLLEGE": "EDUCATION",

    # 9. Relationships
    "WEDDINGS": "RELATIONSHIPS",
    "DIVORCE": "RELATIONSHIPS"
}


df["merged_category"] = (
    df["category"]
    .map(merge_map)
    .fillna(df["category"])
)

print("\nMerged category list:")
print(sorted(df["merged_category"].unique()))

print(
    "\nMerged classes:",
    df["merged_category"].nunique()
)

if df["merged_category"].nunique() != 33:
    raise ValueError(
        f"Expected 33 classes after label merging, "
        f"but got {df['merged_category'].nunique()}."
    )

# ============================================================
# 7. FIXED 15% STRATIFIED HOLDOUT
# ============================================================

indices = range(len(df))

train_idx, test_idx = train_test_split(
    list(indices),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["merged_category"]
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# ============================================================
# 8. SAVE PROCESSED DATA
# ============================================================

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


# ============================================================
# 9. SAVE DATA METADATA
# ============================================================

metadata = {
    "dataset": "News Category Dataset v3",
    "original_classes": int(
        df["category"].nunique()
    ),
    "merged_classes": int(
        df["merged_category"].nunique()
    ),
    "train_rows": int(
        len(train_df)
    ),
    "test_rows": int(
        len(test_df)
    ),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "target": "merged_category",
    "text_feature": "headline + short_description"
}

metadata_path = os.path.join(
    PROCESSED_DIR,
    "dataset_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print("\nPreprocessing completed successfully.")

print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)

print(
    "Test percentage:",
    round(
        len(test_df) / len(df) * 100,
        2
    ),
    "%"
)

print("\nSaved:")
print(train_path)
print(test_path)
print(metadata_path)