"""
Database Initialization Script for GrievanceRoute.
Initializes SQLite schema and seeds initial complaints from cleaned dataset.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grievanceroute.database.repository import GrievanceRepository


def main():
    repo = GrievanceRepository("data/grievance_route.db")
    print("Database schema initialized at data/grievance_route.db")

    csv_path = Path("data/processed/cleaned_complaints.csv")
    if csv_path.exists():
        seeded = repo.seed_initial_complaints(csv_path, max_rows=300)
        print(f"Seeded {seeded} complaint records into database.")
    else:
        print("Note: data/processed/cleaned_complaints.csv not found, skipped initial seeding.")


if __name__ == "__main__":
    main()
