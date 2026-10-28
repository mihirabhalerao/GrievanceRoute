"""
OpenStreetMap-Backed Road Network Graph and A* Routing Service for GrievanceRoute.
Provides caching, bounding-box queries, nearest-node coordinate mapping,
and road-following pathfinding with Euclidean distance lower-bound heuristic.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx

from grievanceroute.ai.search import a_star_search, euclidean_distance_km

logger = logging.getLogger(__name__)

# Default bounding box: Central Pune (Shivajinagar, Deccan, FC Road corridor)
# Format for OSMnx 2.x: (left/min_lon, bottom/min_lat, right/max_lon, top/max_lat)
DEFAULT_PUNE_OSM_BBOX = (73.835, 18.515, 73.855, 18.538)


class OSMRoadGraphService:
    """
    Manages road graph loading from cached OpenStreetMap GraphML data,
    coordinates-to-graph projection, and A* pathfinding.
    Gracefully falls back to the synthetic Pune road network when offline or missing.
    """

    def __init__(
        self,
        cache_file: str | Path = "data/cache/osm_pune_central.graphml",
        fallback_file: str | Path = "data/raw/road_network.json",
        bbox: Tuple[float, float, float, float] = DEFAULT_PUNE_OSM_BBOX,
        allow_download: bool = False,
    ):
        self.cache_file = Path(cache_file)
        self.fallback_file = Path(fallback_file)
        self.bbox = bbox
        self.allow_download = allow_download

        self.graph = nx.Graph()
        self.node_coords: Dict[str, Tuple[float, float]] = {}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self.is_osm: bool = False
        self.source_label: str = "Uninitialized"

        self.load_graph(allow_download=self.allow_download)

    def load_graph(self, allow_download: bool = False) -> bool:
        """
        Load road graph with priority:
        1. Local OSM cache file (GraphML)
        2. Live OSM download via OSMnx (only if allow_download=True and cache missing)
        3. Fallback to synthetic road_network.json
        """
        # 1. Try local cache
        if self.cache_file.exists():
            if self._load_from_graphml(self.cache_file):
                self.is_osm = True
                self.source_label = f"OpenStreetMap (Cached • {self.cache_file.name})"
                return True

        # 2. Try live download if explicitly allowed and cache missing
        if allow_download:
            downloaded = self._download_osm_bbox(self.bbox)
            if downloaded and self._load_from_graphml(self.cache_file):
                self.is_osm = True
                self.source_label = "OpenStreetMap (Live Overpass Download)"
                return True

        # 3. Fallback to bundled synthetic graph
        logger.warning("OSM graph unavailable. Falling back to synthetic road network.")
        return self._load_fallback_graph()

    def _download_osm_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Attempt to fetch road network within bbox via OSMnx and save to cache."""
        try:
            import osmnx as ox
            logger.info("Downloading OSM street network for bbox: %s", bbox)
            g_raw = ox.graph_from_bbox(bbox=bbox, network_type="drive")
            if g_raw and len(g_raw.nodes) > 0:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(g_raw, self.cache_file)
                return True
        except Exception as exc:
            logger.warning("Failed to fetch OSM data: %s", exc)
        return False

    def _load_from_graphml(self, file_path: Path) -> bool:
        """Parse GraphML road graph into internal NetworkX graph with km edge weights."""
        try:
            raw_g = nx.read_graphml(file_path)
            if raw_g.number_of_nodes() == 0:
                return False

            g = nx.Graph()
            node_coords: Dict[str, Tuple[float, float]] = {}
            node_meta: Dict[str, Dict[str, Any]] = {}

            for nid, data in raw_g.nodes(data=True):
                str_nid = str(nid)
                lat = float(data.get("y", data.get("lat", 0.0)))
                lon = float(data.get("x", data.get("lon", 0.0)))
                node_coords[str_nid] = (lat, lon)
                node_meta[str_nid] = {"lat": lat, "lon": lon, "label": f"Node {str_nid}"}
                g.add_node(str_nid, lat=lat, lon=lon)

            for u, v, data in raw_g.edges(data=True):
                str_u, str_v = str(u), str(v)
                # OSM length is in meters; convert to kilometers
                length_m = float(data.get("length", 15.0))
                dist_km = length_m / 1000.0
                road_name = str(data.get("name", "Unnamed Street"))

                if g.has_edge(str_u, str_v):
                    if dist_km < g[str_u][str_v]["distance_km"]:
                        g[str_u][str_v]["distance_km"] = dist_km
                        g[str_u][str_v]["road_name"] = road_name
                else:
                    g.add_edge(str_u, str_v, distance_km=dist_km, road_name=road_name)

            self.graph = g
            self.node_coords = node_coords
            self.node_metadata = node_meta
            return True
        except Exception as exc:
            logger.error("Error reading GraphML: %s", exc)
            return False

    def _load_fallback_graph(self) -> bool:
        """Load the standard synthetic Pune road network."""
        if not self.fallback_file.exists():
            self.is_osm = False
            self.source_label = "Unavailable (Missing files)"
            return False

        try:
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            g = nx.Graph()
            node_coords: Dict[str, Tuple[float, float]] = {}
            node_meta = data.get("nodes", {})

            for nid, n_info in node_meta.items():
                str_nid = str(nid)
                lat = float(n_info["lat"])
                lon = float(n_info["lon"])
                node_coords[str_nid] = (lat, lon)
                g.add_node(str_nid, lat=lat, lon=lon, label=n_info.get("label", str_nid), zone=n_info.get("zone", ""))

            for edge in data.get("edges", []):
                u = str(edge["source"])
                v = str(edge["target"])
                dist = float(edge.get("distance_km", 1.0))
                r_name = edge.get("road_name", "Main Road")
                g.add_edge(u, v, distance_km=dist, road_name=r_name)

            self.graph = g
            self.node_coords = node_coords
            self.node_metadata = node_meta
            self.is_osm = False
            self.source_label = "Fallback Synthetic Road Graph (Offline)"
            return True
        except Exception as exc:
            logger.error("Error loading fallback graph: %s", exc)
            return False

    def find_nearest_node(self, lat: float, lon: float) -> str:
        """Find the closest graph node to given coordinates using Euclidean distance."""
        best_node = None
        best_dist = float("inf")
        for nid, (n_lat, n_lon) in self.node_coords.items():
            d = euclidean_distance_km(lat, lon, n_lat, n_lon)
            if d < best_dist:
                best_dist = d
                best_node = nid

        if best_node is None:
            # Fallback to any node if available
            return next(iter(self.node_coords.keys())) if self.node_coords else ""
        return best_node

    def compute_route(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> Dict[str, Any]:
        """
        Map origin and destination to nearest road network nodes and run A* search.
        Returns detailed path coordinates, road distance, straight-line distance,
        node count, and search execution metrics.
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "success": False,
                "error": "Road network graph is empty.",
                "path_nodes": [],
                "path_coords": [],
                "road_distance_km": 0.0,
                "straight_line_km": 0.0,
                "is_osm": self.is_osm,
                "source_label": self.source_label,
            }

        start_node = self.find_nearest_node(start_lat, start_lon)
        goal_node = self.find_nearest_node(goal_lat, goal_lon)

        straight_line_km = euclidean_distance_km(start_lat, start_lon, goal_lat, goal_lon)

        if start_node == goal_node:
            s_lat, s_lon = self.node_coords[start_node]
            return {
                "success": True,
                "start_node": start_node,
                "goal_node": goal_node,
                "path_nodes": [start_node],
                "path_coords": [[s_lat, s_lon]],
                "node_details": [{"node": start_node, "lat": s_lat, "lon": s_lon, "road": "At Destination"}],
                "road_distance_km": 0.0,
                "straight_line_km": round(straight_line_km, 3),
                "detour_ratio": 1.0,
                "expanded_count": 1,
                "runtime_ms": 0.1,
                "is_osm": self.is_osm,
                "source_label": self.source_label,
            }

        # Run A* Search
        search_res = a_star_search(
            self.graph,
            start_node,
            goal_node,
            self.node_coords,
            weight_attr="distance_km",
        )

        if not search_res["success"]:
            return {
                "success": False,
                "error": search_res.get("error", "Destination unreachable via road network"),
                "start_node": start_node,
                "goal_node": goal_node,
                "path_nodes": [],
                "path_coords": [],
                "road_distance_km": float("inf"),
                "straight_line_km": round(straight_line_km, 3),
                "expanded_count": search_res.get("expanded_count", 0),
                "runtime_ms": search_res.get("runtime_ms", 0.0),
                "is_osm": self.is_osm,
                "source_label": self.source_label,
            }

        path = search_res["path"]
        path_coords: List[List[float]] = []
        node_details: List[Dict[str, Any]] = []

        for i, nid in enumerate(path):
            n_lat, n_lon = self.node_coords[nid]
            path_coords.append([n_lat, n_lon])

            step_road = "Origin Node"
            step_dist = 0.0
            if i > 0:
                prev_id = path[i - 1]
                edge_info = self.graph.get_edge_data(prev_id, nid, {})
                step_road = edge_info.get("road_name", "Street Segment")
                step_dist = round(edge_info.get("distance_km", 0.0), 3)

            meta = self.node_metadata.get(nid, {})
            label = meta.get("label", f"Node {nid}")
            node_details.append({
                "step": i + 1,
                "node": nid,
                "label": label,
                "lat": n_lat,
                "lon": n_lon,
                "step_dist_km": step_dist,
                "road": step_road,
            })

        road_dist = search_res["cost"]
        detour_ratio = round(road_dist / max(straight_line_km, 0.001), 2)

        return {
            "success": True,
            "start_node": start_node,
            "goal_node": goal_node,
            "path_nodes": path,
            "path_coords": path_coords,
            "node_details": node_details,
            "road_distance_km": round(road_dist, 3),
            "straight_line_km": round(straight_line_km, 3),
            "detour_ratio": detour_ratio,
            "expanded_count": search_res["expanded_count"],
            "runtime_ms": search_res["runtime_ms"],
            "is_osm": self.is_osm,
            "source_label": self.source_label,
        }

    def verify_optimality_against_dijkstra(self, start_node: str, goal_node: str) -> Dict[str, Any]:
        """
        Compare A* search with NetworkX Dijkstra implementation on the active road graph.
        Returns cost comparison and optimality check.
        """
        if start_node not in self.graph or goal_node not in self.graph:
            return {"optimal": False, "error": "Start or goal not in graph"}

        # Run A*
        astar_res = a_star_search(self.graph, start_node, goal_node, self.node_coords, "distance_km")

        # Run Dijkstra
        try:
            dijkstra_path = nx.dijkstra_path(self.graph, start_node, goal_node, weight="distance_km")
            dijkstra_cost = nx.dijkstra_path_length(self.graph, start_node, goal_node, weight="distance_km")
        except nx.NetworkXNoPath:
            return {
                "astar_success": astar_res["success"],
                "dijkstra_success": False,
                "optimal": not astar_res["success"],
            }

        cost_diff = abs(astar_res["cost"] - round(dijkstra_cost, 3))
        is_optimal = cost_diff <= 0.005  # Within 5 meters / 3-decimal rounding

        return {
            "optimal": is_optimal,
            "astar_cost": astar_res["cost"],
            "dijkstra_cost": round(dijkstra_cost, 3),
            "cost_difference": round(cost_diff, 5),
            "astar_expanded": astar_res["expanded_count"],
            "astar_path_length": len(astar_res["path"]),
            "dijkstra_path_length": len(dijkstra_path),
        }
