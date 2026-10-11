"""
Dataset Validator for GrievanceRoute.
Validates schemas, data types, missing values, duplicates, and target integrity.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


REQUIRED_COLUMNS = [
    "complaint_id",
    "complaint_text",
    "complaint_type",
    "latitude",
    "longitude",
    "zone",
    "timestamp",
    "hour",
    "day_of_week",
    "weather",
    "monsoon",
    "low_lying_area",
    "repeat_complaint",
    "historical_resolution_hours",
    "urgency_score",
    "actual_resolution_hours",
    "status",
    "crew_required",
]

VALID_CATEGORIES = {"pothole", "water leak", "streetlight", "garbage"}
VALID_ZONES = {"Kothrud", "Shivajinagar", "Hadapsar", "Baner", "Swargate", "Viman Nagar"}
PUNE_LAT_BOUNDS = (18.40, 18.65)
PUNE_LON_BOUNDS = (73.70, 74.00)


def validate_complaint_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate complaint DataFrame against specification rules."""
    report: Dict[str, Any] = {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_columns": [],
        "duplicate_id_count": 0,
        "null_counts": {},
        "invalid_categories": [],
        "invalid_urgency_count": 0,
        "out_of_bounds_coords": 0,
        "warnings": [],
        "errors": [],
    }

    # Column check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report["is_valid"] = False
        report["missing_columns"] = missing_cols
        report["errors"].append(f"Missing required columns: {missing_cols}")

    # Duplicate IDs
    if "complaint_id" in df.columns:
        dup_ids = df["complaint_id"].duplicated().sum()
        report["duplicate_id_count"] = int(dup_ids)
        if dup_ids > 0:
            report["warnings"].append(f"Found {dup_ids} duplicate complaint IDs.")

    # Nulls
    nulls = df.isnull().sum().to_dict()
    report["null_counts"] = {k: int(v) for k, v in nulls.items() if v > 0}

    # Categories
    if "complaint_type" in df.columns:
        unique_cats = set(df["complaint_type"].dropna().unique())
        invalid_cats = unique_cats - VALID_CATEGORIES
        if invalid_cats:
            report["is_valid"] = False
            report["invalid_categories"] = list(invalid_cats)
            report["errors"].append(f"Invalid complaint categories: {invalid_cats}")

    # Urgency ranges
    if "urgency_score" in df.columns:
        invalid_urgency = ((df["urgency_score"] < 1) | (df["urgency_score"] > 5)).sum()
        report["invalid_urgency_count"] = int(invalid_urgency)
        if invalid_urgency > 0:
            report["is_valid"] = False
            report["errors"].append(f"{invalid_urgency} records have urgency outside 1-5.")

    # Geographic coordinates
    if "latitude" in df.columns and "longitude" in df.columns:
        oob = (
            (df["latitude"] < PUNE_LAT_BOUNDS[0])
            | (df["latitude"] > PUNE_LAT_BOUNDS[1])
            | (df["longitude"] < PUNE_LON_BOUNDS[0])
            | (df["longitude"] > PUNE_LON_BOUNDS[1])
        ).sum()
        report["out_of_bounds_coords"] = int(oob)
        if oob > 0:
            report["warnings"].append(f"{oob} records fall slightly outside primary bounds.")

    return report


def validate_road_network(road_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate road network node and edge definitions."""
    report = {
        "is_valid": True,
        "node_count": len(road_data.get("nodes", {})),
        "edge_count": len(road_data.get("edges", [])),
        "errors": [],
    }
    if report["node_count"] < 5:
        report["is_valid"] = False
        report["errors"].append("Road network has fewer than 5 nodes.")
    if report["edge_count"] < 5:
        report["is_valid"] = False
        report["errors"].append("Road network has fewer than 5 edges.")
    return report
