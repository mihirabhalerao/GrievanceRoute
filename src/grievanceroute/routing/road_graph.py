"""
Road Network Graph Service for GrievanceRoute.
Loads road_network.json into NetworkX and maps arbitrary GPS coordinates to road nodes.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx

from grievanceroute.ai.search import euclidean_distance_km, a_star_search


class RoadGraphService:
    """Manages Pune road network graph, node lookups, and A* shortest path queries."""

    def __init__(self, json_path: str | Path = "data/raw/road_network.json"):
        self.json_path = Path(json_path)
        self.graph = nx.Graph()
        self.nodes_data: Dict[str, Dict[str, Any]] = {}
        self.node_coords: Dict[str, Tuple[float, float]] = {}
        self._load_network()

    def _load_network(self) -> None:
        if not self.json_path.exists():
            return
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes_data = data.get("nodes", {})
        for node_id, n_info in self.nodes_data.items():
            lat = float(n_info["lat"])
            lon = float(n_info["lon"])
            self.node_coords[node_id] = (lat, lon)
            self.graph.add_node(node_id, lat=lat, lon=lon, label=n_info.get("label", node_id), zone=n_info.get("zone", ""))

        for edge in data.get("edges", []):
            u = edge["source"]
            v = edge["target"]
            dist = float(edge.get("distance_km", 1.0))
            self.graph.add_edge(u, v, distance_km=dist, road_name=edge.get("road_name", ""))

    def find_nearest_node(self, lat: float, lon: float) -> str:
        """Find the nearest graph intersection or depot node to GPS coordinates."""
        best_node = None
        best_dist = float("inf")
        for node_id, (n_lat, n_lon) in self.node_coords.items():
            d = euclidean_distance_km(lat, lon, n_lat, n_lon)
            if d < best_dist:
                best_dist = d
                best_node = node_id
        return best_node or "DEPOT-SHI"

    def compute_shortest_path(self, start_node: str, end_node: str) -> Dict[str, Any]:
        """Compute A* shortest path between two network nodes."""
        return a_star_search(self.graph, start_node, end_node, self.node_coords)
