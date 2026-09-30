import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train_33class.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test_33class.csv")
METADATA_PATH = os.path.join(PROCESSED_DIR, "metadata.json")


def main():
    print("[INFO] Validating preprocessing outputs...")

    required_files = [TRAIN_PATH, TEST_PATH, METADATA_PATH]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected output not found: {path}"
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_columns = [
        "text",
        "authors",
        "year",
        "merged_category"
    ]

    for df_name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"{df_name} is missing columns: {missing}"
            )

        if df.empty:
            raise ValueError(f"{df_name} dataset is empty.")

        if df["merged_category"].nunique() != 33:
            raise ValueError(
                f"{df_name} does not contain all 33 merged classes."
            )

        if df["text"].isna().any():
            raise ValueError(f"{df_name} contains missing text values.")

    if len(train_df) != metadata["train_rows"]:
        raise ValueError("Train row count does not match metadata.")

    if len(test_df) != metadata["test_rows"]:
        raise ValueError("Test row count does not match metadata.")

    actual_test_ratio = len(test_df) / (len(train_df) + len(test_df))

    if abs(actual_test_ratio - 0.15) > 0.001:
        raise ValueError(
            f"Expected approximately 15% test data, got "
            f"{actual_test_ratio:.4f}"
        )

    print("[INFO] Output validation passed.")
    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Test rows : {len(test_df)}")
    print("[INFO] Classes   : 33")
    print("[INFO] Test split: 15%")
    print("[SUCCESS] Lab 5 production data pipeline validation passed.")


if __name__ == "__main__":
    main()
