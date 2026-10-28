"""
Tests for OpenStreetMap-Backed Road Network Routing and A* Pathfinding.
Verifies graph loading, validation, edge distance units, nearest-node projection,
A* vs Dijkstra optimality, disconnected components, and offline fallback behavior.
No live OSM downloads are triggered during test runs.
"""

import tempfile
from pathlib import Path
import networkx as nx
import pytest

from grievanceroute.routing.osm_router import OSMRoadGraphService, DEFAULT_PUNE_OSM_BBOX
from grievanceroute.ai.search import a_star_search, euclidean_distance_km


def test_osm_graph_loading_and_validation():
    """Verify loading cached OSM GraphML and ensuring non-empty validated graph."""
    cache_path = Path("data/cache/osm_pune_central.graphml")
    assert cache_path.exists(), "Cached OSM GraphML file must exist for offline execution."

    router = OSMRoadGraphService(cache_file=cache_path)
    assert router.is_osm is True
    assert "OpenStreetMap" in router.source_label
    assert router.graph.number_of_nodes() > 50
    assert router.graph.number_of_edges() > 50
    assert len(router.node_coords) == router.graph.number_of_nodes()


def test_edge_distance_units_in_kilometers():
    """Verify edge weights are stored in kilometers and are strictly positive."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    for u, v, data in router.graph.edges(data=True):
        dist_km = data.get("distance_km")
        assert dist_km is not None
        assert isinstance(dist_km, float)
        assert dist_km > 0.0
        # In central Pune city corridor, single street blocks are typically < 5 km
        assert dist_km < 10.0


def test_nearest_node_mapping():
    """Verify arbitrary GPS coordinates map to closest valid road network node."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    # Query location near Shivajinagar / FC Road
    query_lat, query_lon = 18.5280, 73.8450
    nearest = router.find_nearest_node(query_lat, query_lon)

    assert nearest in router.node_coords
    node_lat, node_lon = router.node_coords[nearest]
    # Projected distance to nearest node in dense urban area should be small (< 1 km)
    dist = euclidean_distance_km(query_lat, query_lon, node_lat, node_lon)
    assert dist < 1.0


def test_a_star_optimality_versus_dijkstra():
    """
    Verify A* with Euclidean straight-line distance heuristic produces the exact same
    optimal shortest path cost as Dijkstra's algorithm across multiple node pairs.
    """
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    nodes = list(router.graph.nodes)

    # Test several distinct pairs across the network
    test_pairs = [
        (nodes[0], nodes[25]),
        (nodes[10], nodes[50]),
        (nodes[20], nodes[80]),
    ]

    for start, goal in test_pairs:
        # A*
        astar_res = a_star_search(router.graph, start, goal, router.node_coords, "distance_km")
        if astar_res["success"]:
            # Dijkstra ground truth
            dijkstra_cost = nx.dijkstra_path_length(router.graph, start, goal, weight="distance_km")
            # Cost difference within 3-decimal rounding tolerance
            assert abs(astar_res["cost"] - round(dijkstra_cost, 3)) <= 0.005
            assert astar_res["cost"] >= 0.0


def test_route_computation_and_distance():
    """Verify compute_route returns valid path coordinates, road distance, and metrics."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")

    # Start: Deccan Gymkhana area (18.5170, 73.8410)
    # End: Shivajinagar area (18.5314, 73.8446)
    res = router.compute_route(18.5170, 73.8410, 18.5314, 73.8446)

    assert res["success"] is True
    assert res["is_osm"] is True
    assert res["road_distance_km"] > 0.0
    assert res["straight_line_km"] > 0.0
    # Road distance is always >= straight-line distance
    assert res["road_distance_km"] >= res["straight_line_km"] * 0.95
    assert len(res["path_nodes"]) >= 2
    assert len(res["path_coords"]) == len(res["path_nodes"])
    assert res["detour_ratio"] >= 1.0


def test_identical_origin_and_destination():
    """Verify route computation when start and goal coordinates are identical."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    res = router.compute_route(18.5250, 73.8420, 18.5250, 73.8420)

    assert res["success"] is True
    assert res["road_distance_km"] == 0.0
    assert len(res["path_nodes"]) == 1
    assert len(res["path_coords"]) == 1


def test_disconnected_destination_handling():
    """Verify disconnected nodes return success=False and infinite cost gracefully."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Create small disconnected graph
        g = nx.Graph()
        g.add_node("A", lat=18.52, lon=73.84)
        g.add_node("B", lat=18.53, lon=73.85)
        g.add_node("ISLAND", lat=18.60, lon=73.90)
        g.add_edge("A", "B", distance_km=1.5, road_name="Main Rd")

        graphml_path = Path(tmpdir) / "disconnected.graphml"
        nx.write_graphml(g, graphml_path)

        router = OSMRoadGraphService(cache_file=graphml_path)
        res = a_star_search(router.graph, "A", "ISLAND", router.node_coords)

        assert res["success"] is False
        assert res["path"] == []
        assert res["cost"] == float("inf")


def test_fallback_behavior_when_cache_missing():
    """Verify fallback to synthetic road_network.json when OSM cache is missing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        non_existent_cache = Path(tmpdir) / "non_existent.graphml"
        router = OSMRoadGraphService(
            cache_file=non_existent_cache,
            fallback_file="data/raw/road_network.json",
        )

        assert router.is_osm is False
        assert "Fallback Synthetic Road Graph" in router.source_label
        assert router.graph.number_of_nodes() > 0

        # Verify A* routing still functions on fallback graph
        res = router.compute_route(18.5074, 73.8077, 18.5314, 73.8446)
        assert res["success"] is True
        assert res["is_osm"] is False
        assert res["road_distance_km"] > 0.0
