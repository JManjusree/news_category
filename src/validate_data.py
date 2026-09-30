import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "News_Category_Dataset_v3.json")

REQUIRED_COLUMNS = ["headline", "short_description", "authors", "date", "category"]


def main():
    print("[INFO] Validating raw News Category dataset...")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    df = pd.read_json(RAW_PATH, lines=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["category"].isna().any():
        raise ValueError("Category contains missing values.")

    if df["category"].nunique() < 2:
        raise ValueError("Dataset must contain at least two classes.")

    print("[INFO] Raw validation passed.")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Original classes: {df['category'].nunique()}")


if __name__ == "__main__":
    main()
