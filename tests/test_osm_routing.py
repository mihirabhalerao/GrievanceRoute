"""
Tests for OpenStreetMap-Backed Road Network Routing and A* Pathfinding.
Verifies graph loading, validation, edge distance units, nearest-node projection,
curved geometry extraction, one-way street directionality, A* vs Dijkstra optimality,
disconnected components, and offline fallback behavior.
No live OSM downloads are triggered during test runs.
"""

import tempfile
from pathlib import Path
import networkx as nx
import pytest

from grievanceroute.routing.osm_router import (
    OSMRoadGraphService,
    DEFAULT_PUNE_OSM_BBOX,
    parse_wkt_linestring,
)
from grievanceroute.ai.search import a_star_search, euclidean_distance_km


def test_wkt_linestring_parsing():
    """Verify parsing WKT LINESTRING into (latitude, longitude) coordinate tuples."""
    sample = "LINESTRING (73.8392602 18.5160393, 73.8394647 18.5160205, 73.8398983 18.5159806)"
    coords = parse_wkt_linestring(sample)
    assert len(coords) == 3
    # Check that lat is ~18.51 and lon is ~73.83
    assert abs(coords[0][0] - 18.5160393) < 1e-6
    assert abs(coords[0][1] - 73.8392602) < 1e-6


def test_osm_graph_loading_and_validation():
    """Verify loading cached OSM GraphML and ensuring non-empty validated directed graph."""
    cache_path = Path("data/cache/osm_pune_central.graphml")
    assert cache_path.exists(), "Cached OSM GraphML file must exist for offline execution."

    router = OSMRoadGraphService(cache_file=cache_path)
    assert router.is_osm is True
    assert "OpenStreetMap" in router.source_label
    assert router.graph.number_of_nodes() > 50
    assert router.graph.number_of_edges() > 50
    assert len(router.node_coords) == router.graph.number_of_nodes()
    assert router.graph.is_directed() is True


def test_edge_distance_units_in_kilometers():
    """Verify edge weights are stored in kilometers and are strictly positive."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    for u, v, data in router.graph.edges(data=True):
        dist_km = data.get("distance_km")
        assert dist_km is not None
        assert isinstance(dist_km, float)
        assert dist_km > 0.0
        assert dist_km < 10.0


def test_nearest_node_mapping():
    """Verify arbitrary GPS coordinates map to closest valid road network node."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    query_lat, query_lon = 18.5280, 73.8450
    nearest = router.find_nearest_node(query_lat, query_lon)

    assert nearest in router.node_coords
    node_lat, node_lon = router.node_coords[nearest]
    dist = euclidean_distance_km(query_lat, query_lon, node_lat, node_lon)
    assert dist < 1.0


def test_a_star_optimality_versus_dijkstra():
    """
    Verify A* with Euclidean straight-line distance heuristic produces the exact same
    optimal shortest path cost as Dijkstra's algorithm across multiple node pairs on directed graph.
    """
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    nodes = list(router.graph.nodes)

    test_pairs = [
        (nodes[0], nodes[25]),
        (nodes[10], nodes[50]),
        (nodes[20], nodes[80]),
    ]

    for start, goal in test_pairs:
        astar_res = a_star_search(router.graph, start, goal, router.node_coords, "distance_km")
        if astar_res["success"]:
            dijkstra_cost = nx.dijkstra_path_length(router.graph, start, goal, weight="distance_km")
            assert abs(astar_res["cost"] - round(dijkstra_cost, 3)) <= 0.005
            assert astar_res["cost"] >= 0.0


def test_curved_road_geometry_extraction():
    """Verify that route coordinate extraction includes intermediate curve geometry points."""
    router = OSMRoadGraphService(cache_file="data/cache/osm_pune_central.graphml")
    res = router.compute_route(18.5170, 73.8410, 18.5314, 73.8446)

    assert res["success"] is True
    assert res["is_osm"] is True
    assert len(res["path_nodes"]) >= 2
    # Because curved edges have intermediate points, total polyline coordinates must be >= path node count
    assert len(res["path_coords"]) >= len(res["path_nodes"])
    # Verify each coordinate point is a [lat, lon] pair with valid ranges
    for pt in res["path_coords"]:
        assert len(pt) == 2
        assert 18.40 <= pt[0] <= 18.65
        assert 73.70 <= pt[1] <= 74.05


def test_one_way_street_directionality():
    """Verify that directed edges prevent illegal travel against one-way traffic."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Create small directed graph: A -> B only (one-way street)
        g = nx.DiGraph()
        g.add_node("A", lat=18.520, lon=73.840)
        g.add_node("B", lat=18.525, lon=73.845)
        g.add_edge("A", "B", distance_km=0.8, road_name="One Way Ave")

        graphml_path = Path(tmpdir) / "oneway.graphml"
        nx.write_graphml(g, graphml_path)

        router = OSMRoadGraphService(cache_file=graphml_path)

        # Route with traffic direction: A -> B
        res_forward = router.compute_route(18.520, 73.840, 18.525, 73.845)
        assert res_forward["success"] is True
        assert res_forward["road_distance_km"] == 0.8

        # Route against traffic direction: B -> A
        res_reverse = a_star_search(router.graph, "B", "A", router.node_coords)
        assert res_reverse["success"] is False


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
        g = nx.DiGraph()
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
