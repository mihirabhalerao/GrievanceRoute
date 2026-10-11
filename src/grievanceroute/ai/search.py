"""
Heuristic Road Pathfinding for GrievanceRoute.
Implements A* Search over road network graphs with an admissible Euclidean distance lower-bound heuristic.
"""

from __future__ import annotations
import heapq
import math
import time
from typing import Dict, List, Tuple, Any

import networkx as nx


def euclidean_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute straight-line Euclidean distance lower bound in kilometers."""
    lat_km = (lat2 - lat1) * 110.574
    lon_km = (lon2 - lon1) * 111.320 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.sqrt(lat_km ** 2 + lon_km ** 2)


def a_star_search(
    graph: nx.Graph,
    start: str,
    goal: str,
    node_coords: Dict[str, Tuple[float, float]],
    weight_attr: str = "distance_km",
) -> Dict[str, Any]:
    """
    A* algorithm over weighted road graph.
    Evaluates f(n) = g(n) + h(n), where h(n) is admissible straight-line distance.
    Returns optimal path, path cost, expanded node count, and runtime.
    """
    start_time = time.time()
    if start not in graph or goal not in graph:
        return {
            "success": False,
            "error": f"Start '{start}' or Goal '{goal}' not found in road graph.",
            "path": [],
            "cost": float("inf"),
            "expanded_count": 0,
            "visited_nodes": [],
            "runtime_ms": 0.0,
        }

    goal_lat, goal_lon = node_coords[goal]

    def heuristic(node: str) -> float:
        n_lat, n_lon = node_coords[node]
        return euclidean_distance_km(n_lat, n_lon, goal_lat, goal_lon)

    counter = 0
    start_h = heuristic(start)
    open_set: List[Tuple[float, int, float, str, List[str]]] = [
        (start_h, counter, 0.0, start, [start])
    ]
    g_scores: Dict[str, float] = {start: 0.0}
    visited_nodes: List[str] = []
    expanded_count = 0

    while open_set:
        f, _, g, current, path = heapq.heappop(open_set)

        if current in visited_nodes:
            continue
        visited_nodes.append(current)
        expanded_count += 1

        if current == goal:
            duration = (time.time() - start_time) * 1000
            return {
                "success": True,
                "path": path,
                "cost": round(g, 3),
                "expanded_count": expanded_count,
                "visited_nodes": visited_nodes,
                "runtime_ms": round(duration, 3),
            }

        for neighbor in graph.neighbors(current):
            edge_weight = graph[current][neighbor].get(weight_attr, 1.0)
            tentative_g = g + edge_weight

            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                h = heuristic(neighbor)
                counter += 1
                heapq.heappush(
                    open_set,
                    (tentative_g + h, counter, tentative_g, neighbor, path + [neighbor])
                )

    duration = (time.time() - start_time) * 1000
    return {
        "success": False,
        "error": "Goal unreachable",
        "path": [],
        "cost": float("inf"),
        "expanded_count": expanded_count,
        "visited_nodes": visited_nodes,
        "runtime_ms": round(duration, 3),
    }
