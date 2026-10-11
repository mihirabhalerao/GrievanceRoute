"""
Unsupervised Spatial Clustering for GrievanceRoute.
Implements K-Means (projected metric coordinates in km) and DBSCAN density-based hotspot detection.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from grievanceroute.data.feature_engineering import project_coordinates_to_km, PUNE_CENTER_LAT, PUNE_CENTER_LON


def run_kmeans_hotspots(
    df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run K-Means clustering on projected coordinates (km)."""
    coords_km = project_coordinates_to_km(df["latitude"], df["longitude"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords_km)

    # Convert cluster centers back to lat/lon for mapping
    centers_km = kmeans.cluster_centers_
    lat_deg_per_km = 1.0 / 110.574
    lon_deg_per_km = 1.0 / (111.320 * math.cos(math.radians(PUNE_CENTER_LAT)))

    center_lats = PUNE_CENTER_LAT + centers_km[:, 1] * lat_deg_per_km
    center_lons = PUNE_CENTER_LON + centers_km[:, 0] * lon_deg_per_km

    # Silhouette score
    sample_size = min(2000, len(coords_km))
    sil_score = silhouette_score(coords_km, labels, sample_size=sample_size, random_state=random_state)

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip([int(u) for u in unique], [int(c) for c in counts]))

    centers_list = [
        {
            "cluster_id": i,
            "lat": round(float(center_lats[i]), 6),
            "lon": round(float(center_lons[i]), 6),
            "size": cluster_sizes.get(i, 0),
        }
        for i in range(n_clusters)
    ]

    summary = {
        "n_clusters": n_clusters,
        "inertia": round(float(kmeans.inertia_), 2),
        "silhouette_score": round(float(sil_score), 4),
        "cluster_sizes": cluster_sizes,
        "cluster_centers": centers_list,
    }

    return labels, summary


def run_dbscan_hotspots(
    df: pd.DataFrame,
    eps_km: float = 1.2,
    min_samples: int = 25,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run DBSCAN density clustering on projected coordinates to discover high-density complaint clusters & noise."""
    coords_km = project_coordinates_to_km(df["latitude"], df["longitude"])
    dbscan = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = dbscan.fit_predict(coords_km)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    noise_fraction = round(noise_count / len(labels), 4)

    summary = {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_fraction": noise_fraction,
        "eps_km": eps_km,
        "min_samples": min_samples,
    }

    return labels, summary
