"""
Preprocessing Script for GrievanceRoute.
Cleans raw data, deduplicates records, normalizes features, and saves processed CSV.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grievanceroute.data.preprocessor import clean_and_preprocess_complaints


def main():
    raw_path = Path("data/raw/pune_civic_complaints.csv")
    out_path = Path("data/processed/cleaned_complaints.csv")

    if not raw_path.exists():
        print(f"Error: {raw_path} not found.")
        sys.exit(1)

    print(f"Loading raw data from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    df_clean, summary = clean_and_preprocess_complaints(df_raw, save_path=out_path)

    print("Preprocessing completed!")
    print(f"Initial rows: {summary['initial_rows']}")
    print(f"Duplicates removed: {summary['duplicates_removed']}")
    print(f"Weather nulls imputed: {summary['weather_nulls_imputed']}")
    print(f"Final cleaned rows: {summary['final_cleaned_rows']}")
    print(f"Saved processed dataset to {out_path}")


if __name__ == "__main__":
    main()
