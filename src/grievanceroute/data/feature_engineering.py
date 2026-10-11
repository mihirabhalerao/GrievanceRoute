"""
Feature Engineering and Coordinate Projection for GrievanceRoute.
Includes projected metric coordinates for Pune, TF-IDF transformations,
and scikit-learn compatible ColumnTransformers.
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


PUNE_CENTER_LAT = 18.5204
PUNE_CENTER_LON = 73.8567


def project_coordinates_to_km(
    latitudes: np.ndarray | pd.Series,
    longitudes: np.ndarray | pd.Series,
    center_lat: float = PUNE_CENTER_LAT,
    center_lon: float = PUNE_CENTER_LON,
) -> np.ndarray:
    """
    Project latitude and longitude to local planar coordinates (x_km, y_km)
    centered around Pune using equirectangular approximation.
    Ensures Euclidean distance in clustering and routing algorithms is in metric kilometers.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)

    # 1 degree lat approx 110.574 km; 1 degree lon approx 111.320 * cos(lat) km
    lat_km_per_deg = 110.574
    lon_km_per_deg = 111.320 * math.cos(math.radians(center_lat))

    y_km = (lats - center_lat) * lat_km_per_deg
    x_km = (lons - center_lon) * lon_km_per_deg

    return np.column_stack([x_km, y_km])


def build_text_vectorizer(
    max_features: int = 2500,
    ngram_range: Tuple[int, int] = (1, 2)
) -> TfidfVectorizer:
    """Create a configured TF-IDF vectorizer for complaints."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def build_tabular_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for tabular context features."""
    numeric_features = ["latitude", "longitude", "hour", "day_of_week", "historical_resolution_hours"]
    categorical_features = ["zone", "weather"]
    binary_features = ["monsoon", "low_lying_area", "repeat_complaint"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop"
    )
    return preprocessor


def split_data_stratified(
    df: pd.DataFrame,
    target_col: str = "complaint_type",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform reproducible stratified train-test split."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col] if target_col in df.columns else None
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
