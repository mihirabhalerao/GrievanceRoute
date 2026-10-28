"""
OpenStreetMap-Backed Road Network Graph and A* Routing Service for GrievanceRoute.
Provides caching, bounding-box queries, nearest-node coordinate mapping,
directed one-way street navigation, and curved road geometry reconstruction
with an admissible Euclidean distance lower-bound heuristic.
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


def parse_wkt_linestring(geom_str: str) -> List[Tuple[float, float]]:
    """
    Parse a WKT LINESTRING like 'LINESTRING (lon1 lat1, lon2 lat2, ...)'
    into a list of (latitude, longitude) coordinate tuples.
    """
    if not geom_str or not geom_str.startswith("LINESTRING"):
        return []
    coords: List[Tuple[float, float]] = []
    try:
        content = geom_str[geom_str.find("(") + 1 : geom_str.rfind(")")]
        parts = content.split(",")
        for part in parts:
            tokens = part.strip().split()
            if len(tokens) >= 2:
                lon = float(tokens[0])
                lat = float(tokens[1])
                coords.append((lat, lon))
    except Exception as exc:
        logger.debug("Failed to parse WKT linestring: %s", exc)
    return coords


class OSMRoadGraphService:
    """
    Manages road graph loading from cached OpenStreetMap GraphML data,
    coordinates-to-graph projection, directed edge handling, curved road geometry
    reconstruction, and A* pathfinding.
    Gracefully falls back to the synthetic Pune road network when offline.
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

        self.graph = nx.DiGraph()
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
        """
        Parse GraphML into directed NetworkX DiGraph, retaining one-way directions,
        kilometers edge weights, and curved road geometries.
        """
        try:
            raw_g = nx.read_graphml(file_path)
            if raw_g.number_of_nodes() == 0:
                return False

            g = nx.DiGraph()
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
                if "length" in data:
                    dist_km = float(data["length"]) / 1000.0
                elif "distance_km" in data:
                    dist_km = float(data["distance_km"])
                else:
                    dist_km = 0.015

                road_name = str(data.get("name", data.get("road_name", "Unnamed Street")))
                geom_str = str(data.get("geometry", ""))
                geom_pts = parse_wkt_linestring(geom_str)

                # Directed edge handling
                if g.has_edge(str_u, str_v):
                    if dist_km < g[str_u][str_v]["distance_km"]:
                        g[str_u][str_v]["distance_km"] = dist_km
                        g[str_u][str_v]["road_name"] = road_name
                        g[str_u][str_v]["geometry_pts"] = geom_pts
                else:
                    g.add_edge(str_u, str_v, distance_km=dist_km, road_name=road_name, geometry_pts=geom_pts)

            self.graph = g
            self.node_coords = node_coords
            self.node_metadata = node_meta
            return True
        except Exception as exc:
            logger.error("Error reading GraphML: %s", exc)
            return False

    def _load_fallback_graph(self) -> bool:
        """Load the standard synthetic Pune road network into directed DiGraph."""
        if not self.fallback_file.exists():
            self.is_osm = False
            self.source_label = "Unavailable (Missing files)"
            return False

        try:
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            g = nx.DiGraph()
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
                # Fallback graph is bidirectional
                g.add_edge(u, v, distance_km=dist, road_name=r_name, geometry_pts=[])
                g.add_edge(v, u, distance_km=dist, road_name=r_name, geometry_pts=[])

            self.graph = g
            self.node_coords = node_coords
            self.node_metadata = node_meta
            self.is_osm = False
            self.source_label = "Fallback Synthetic Road Graph (Offline)"
            return True
        except Exception as exc:
            logger.error("Error loading fallback graph: %s", exc)
            return False

    def find_nearest_node(self, lat: float, lon: float, require_outbound: bool = True) -> str:
        """Find the closest graph node to given coordinates using Euclidean distance."""
        best_node = None
        best_dist = float("inf")
        for nid, (n_lat, n_lon) in self.node_coords.items():
            if require_outbound and self.graph.out_degree(nid) == 0:
                continue
            d = euclidean_distance_km(lat, lon, n_lat, n_lon)
            if d < best_dist:
                best_dist = d
                best_node = nid

        if best_node is None and self.node_coords:
            best_node = next(iter(self.node_coords.keys()))
        return best_node or ""

    def _extract_road_following_polyline(self, path: List[str]) -> List[List[float]]:
        """
        Reconstruct the full continuous road-following polyline for the ordered node path.
        Concatenates intermediate curve points from edge geometries in the correct travel order.
        """
        if not path:
            return []

        if len(path) == 1:
            n_lat, n_lon = self.node_coords[path[0]]
            return [[n_lat, n_lon]]

        full_coords: List[List[float]] = [[self.node_coords[path[0]][0], self.node_coords[path[0]][1]]]

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            u_lat, u_lon = self.node_coords[u]
            v_lat, v_lon = self.node_coords[v]

            edge_data = self.graph.get_edge_data(u, v, {})
            pts = edge_data.get("geometry_pts", [])

            if pts and len(pts) > 1:
                # Determine travel direction along geometry points
                d_first = euclidean_distance_km(pts[0][0], pts[0][1], u_lat, u_lon)
                d_last = euclidean_distance_km(pts[-1][0], pts[-1][1], u_lat, u_lon)
                ordered_pts = pts if d_first <= d_last else pts[::-1]

                for p in ordered_pts:
                    pt_list = [p[0], p[1]]
                    # Avoid duplicate sequential coordinates
                    if pt_list != full_coords[-1]:
                        full_coords.append(pt_list)
            else:
                v_pt = [v_lat, v_lon]
                if v_pt != full_coords[-1]:
                    full_coords.append(v_pt)

        return full_coords

    def compute_route(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> Dict[str, Any]:
        """
        Map origin and destination to nearest road nodes, execute A* search over directed graph,
        and reconstruct the complete curved road geometry.
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

        start_node = self.find_nearest_node(start_lat, start_lon, require_outbound=True)
        goal_node = self.find_nearest_node(goal_lat, goal_lon, require_outbound=False)

        straight_line_km = euclidean_distance_km(start_lat, start_lon, goal_lat, goal_lon)

        if start_node == goal_node:
            s_lat, s_lon = self.node_coords[start_node]
            return {
                "success": True,
                "start_node": start_node,
                "goal_node": goal_node,
                "path_nodes": [start_node],
                "path_coords": [[s_lat, s_lon]],
                "node_details": [{"step": 1, "node": start_node, "lat": s_lat, "lon": s_lon, "step_dist_km": 0.0, "road": "At Destination"}],
                "road_distance_km": 0.0,
                "straight_line_km": round(straight_line_km, 3),
                "detour_ratio": 1.0,
                "expanded_count": 1,
                "runtime_ms": 0.1,
                "is_osm": self.is_osm,
                "source_label": self.source_label,
            }

        # Run A* Search over directed road graph
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
                "error": search_res.get("error", "Destination unreachable via road network. Check one-way directions or boundary barriers."),
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
        # Extract full high-fidelity road polyline including curved geometry points
        path_coords = self._extract_road_following_polyline(path)

        # Build step-by-step turn itinerary
        node_details: List[Dict[str, Any]] = []
        cum_dist = 0.0

        for i, nid in enumerate(path):
            n_lat, n_lon = self.node_coords[nid]
            step_road = "Origin Node"
            step_dist = 0.0

            if i > 0:
                prev_id = path[i - 1]
                edge_info = self.graph.get_edge_data(prev_id, nid, {})
                step_road = edge_info.get("road_name", "Street Segment")
                step_dist = round(edge_info.get("distance_km", 0.0), 3)
                cum_dist += step_dist

            meta = self.node_metadata.get(nid, {})
            label = meta.get("label", f"Node {nid}")
            node_details.append({
                "step": i + 1,
                "node": nid,
                "label": label,
                "lat": n_lat,
                "lon": n_lon,
                "step_dist_km": step_dist,
                "cum_dist_km": round(cum_dist, 3),
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
        is_optimal = cost_diff <= 0.005  # Within 5 meters / rounding

        return {
            "optimal": is_optimal,
            "astar_cost": astar_res["cost"],
            "dijkstra_cost": round(dijkstra_cost, 3),
            "cost_difference": round(cost_diff, 5),
            "astar_expanded": astar_res["expanded_count"],
            "astar_path_length": len(astar_res["path"]),
            "dijkstra_path_length": len(dijkstra_path),
        }
