"""
Dataset Generation Script for GrievanceRoute.
Generates 8,000 synthetic Pune civic complaints, crews, and road network.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from grievanceroute.data.generator import generate_full_dataset


def main():
    print("Generating GrievanceRoute synthetic dataset...")
    df, crews, road = generate_full_dataset(
        target_count=8000,
        seed=42,
        raw_csv_path="data/raw/pune_civic_complaints.csv",
        crews_csv_path="data/raw/crews.csv",
        road_json_path="data/raw/road_network.json",
    )
    print(f"Generated {len(df)} complaint records.")
    print(f"Generated {len(crews)} crew teams.")
    print(f"Generated road network with {len(road['nodes'])} nodes and {len(road['edges'])} directed edges.")
    print("Saved files to data/raw/")


if __name__ == "__main__":
    main()
