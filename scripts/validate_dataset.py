"""
Dataset Validation Script for GrievanceRoute.
Validates schemas, data types, value boundaries, and integrity.
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grievanceroute.data.validator import validate_complaint_dataframe, validate_road_network


def main():
    raw_path = Path("data/raw/pune_civic_complaints.csv")
    road_path = Path("data/raw/road_network.json")

    if not raw_path.exists():
        print(f"Error: {raw_path} does not exist. Run scripts/generate_dataset.py first.")
        sys.exit(1)

    print("Validating raw complaints dataset...")
    df = pd.read_csv(raw_path)
    report = validate_complaint_dataframe(df)

    print(f"Row count: {report['row_count']}")
    print(f"Columns: {report['column_count']}")
    print(f"Valid schema: {report['is_valid']}")
    print(f"Duplicate IDs: {report['duplicate_id_count']}")
    print(f"Null counts: {report['null_counts']}")
    print(f"Warnings: {report['warnings']}")
    print(f"Errors: {report['errors']}")

    if road_path.exists():
        with open(road_path, "r", encoding="utf-8") as f:
            road_data = json.load(f)
        road_report = validate_road_network(road_data)
        print(f"Road network valid: {road_report['is_valid']} ({road_report['node_count']} nodes, {road_report['edge_count']} edges)")


if __name__ == "__main__":
    main()
