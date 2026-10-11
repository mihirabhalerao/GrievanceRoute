"""
Tests for Retained Heuristic Search and Local Optimization Algorithms.
Focuses on A* road network pathfinding, 2-Opt Hill Climbing, and Simulated Annealing tour optimization.
"""

import networkx as nx
import pytest

from grievanceroute.ai.search import a_star_search, euclidean_distance_km
from grievanceroute.ai.local_search import hill_climbing_permutation, simulated_annealing_permutation


def test_a_star_optimality_on_known_graph():
    """Verify A* finds the optimal shortest path on a known triangle graph."""
    g = nx.Graph()
    # Path A-B is 10 km direct. Path A-C is 3 km, C-B is 4 km. Path via C is 7 km.
    g.add_edge("A", "B", distance_km=10.0)
    g.add_edge("A", "C", distance_km=3.0)
    g.add_edge("C", "B", distance_km=4.0)

    coords = {"A": (18.50, 73.80), "B": (18.52, 73.85), "C": (18.51, 73.82)}
    res = a_star_search(g, "A", "B", coords)
    assert res["success"] is True
    assert res["path"] == ["A", "C", "B"]
    assert res["cost"] == 7.0
    assert res["expanded_count"] > 0


def test_a_star_unreachable_node():
    """Verify A* handles disconnected graph components gracefully."""
    g = nx.Graph()
    g.add_edge("A", "B", distance_km=2.0)
    g.add_node("C")
    coords = {"A": (18.50, 73.80), "B": (18.51, 73.81), "C": (18.60, 73.90)}
    res = a_star_search(g, "A", "C", coords)
    assert res["success"] is False
    assert res["path"] == []
    assert res["cost"] == float("inf")


def test_hill_climbing_2opt_improvement():
    """Verify 2-Opt Hill Climbing improves tour permutation distance."""
    # A suboptimal tour
    initial_seq = [0, 4, 1, 5, 2, 3]
    # Simple metric: sum of consecutive differences
    def cost_fn(seq):
        return sum(abs(seq[i] - seq[i + 1]) for i in range(len(seq) - 1))

    res = hill_climbing_permutation(initial_seq, cost_fn, max_evaluations=100)
    assert res["final_cost"] <= res["initial_cost"]


def test_simulated_annealing_tour_optimizer():
    """Verify Simulated Annealing achieves lower or equal cost compared to initial order."""
    initial_seq = [5, 1, 4, 0, 3, 2]
    def cost_fn(seq):
        return sum(abs(seq[i] - seq[i + 1]) for i in range(len(seq) - 1))

    res = simulated_annealing_permutation(initial_seq, cost_fn, max_iterations=60, seed=42)
    assert res["final_cost"] <= res["initial_cost"]
    assert len(res["best_solution"]) == len(initial_seq)
