"""
Tests for AI Search Algorithms: A*, Best-First, Local Search, Water-Jug, 8-Puzzle, AO*, Minimax.
"""

import networkx as nx
import pytest

from grievanceroute.ai.search import a_star_search, greedy_best_first_search, euclidean_distance_km
from grievanceroute.ai.local_search import hill_climbing_permutation, simulated_annealing_permutation
from grievanceroute.ai.classic import (
    solve_water_jug,
    solve_8_puzzle,
    is_8_puzzle_solvable,
    manhattan_distance_8_puzzle,
)
from grievanceroute.ai.game_and_or import solve_ao_star, minimax_alpha_beta


def test_a_star_optimality_on_known_graph():
    """Verify A* finds the optimal shortest path on a small triangle graph."""
    g = nx.Graph()
    # Path A-B is 10 km. Path A-C is 3 km, C-B is 4 km. Total via C is 7 km.
    g.add_edge("A", "B", distance_km=10.0)
    g.add_edge("A", "C", distance_km=3.0)
    g.add_edge("C", "B", distance_km=4.0)

    coords = {"A": (18.50, 73.80), "B": (18.52, 73.85), "C": (18.51, 73.82)}
    res = a_star_search(g, "A", "B", coords)
    assert res["success"] is True
    assert res["path"] == ["A", "C", "B"]
    assert res["cost"] == 7.0


def test_water_jug_solvable_and_unsolvable():
    """Verify water-jug solver handles solvable targets and flags impossible GCD cases."""
    # (4, 3) -> target 2 is solvable
    res_solv = solve_water_jug(cap1=4, cap2=3, target=2)
    assert res_solv["solvable"] is True
    assert res_solv["step_count"] > 0

    # (6, 4) -> target 3 is impossible because gcd(6, 4) = 2 does not divide 3
    res_unsolv = solve_water_jug(cap1=6, cap2=4, target=3)
    assert res_unsolv["solvable"] is False
    assert "not a multiple of gcd" in res_unsolv["reason"]


def test_8_puzzle_solvable_and_unsolvable():
    """Verify 8-puzzle parity check and A* solution path."""
    # Solvable state (2 moves from goal)
    state_solvable = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert is_8_puzzle_solvable(state_solvable) is True
    sol = solve_8_puzzle(state_solvable, algorithm="A*")
    assert sol["solvable"] is True
    assert sol["step_count"] == 1

    # Unsolvable state (swapped 7 and 8 has odd inversion count)
    state_unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)
    assert is_8_puzzle_solvable(state_unsolvable) is False
    unsol = solve_8_puzzle(state_unsolvable, algorithm="A*")
    assert unsol["solvable"] is False
    assert "inversion count" in unsol["reason"]


def test_ao_star_solver():
    """Verify AO* bottom-up cost evaluation and solution extraction."""
    graph = {
        "A": {"h": 5.0, "branches": [
            {"type": "OR", "children": ["B"], "edge_cost": 1.0},
            {"type": "AND", "children": ["C", "D"], "edge_cost": 1.0},
        ]},
        "B": {"h": 8.0, "branches": []},
        "C": {"h": 1.0, "branches": []},
        "D": {"h": 2.0, "branches": []},
    }
    res = solve_ao_star(graph, root_name="A")
    assert res["solved"] is True
    # B costs 1 + 8 = 9. C+D costs 1 + 1 + 2 = 4. AO* picks AND branch (cost 4.0)
    assert res["total_cost"] == 4.0


def test_minimax_alpha_beta():
    """Verify minimax evaluation and pruning on adversarial tree."""
    tree = {
        "root": {"children": ["n1", "n2"]},
        "n1": {"children": ["n11", "n12"]},
        "n2": {"children": ["n21", "n22"]},
        "n11": {"value": 3.0},
        "n12": {"value": 5.0},
        "n21": {"value": 6.0},
        "n22": {"value": 9.0},
    }
    stats = {"leaves_evaluated": 0, "pruned_branches": 0}
    val, best_move = minimax_alpha_beta(tree, "root", depth=2, alpha=float("-inf"), beta=float("inf"), is_maximizing=True, stats=stats)
    # Minimax at n1: min(3, 5) = 3. Minimax at n2: min(6, 9) = 6. Max at root: max(3, 6) = 6.
    assert val == 6.0
    assert best_move == "n2"


def test_simulated_annealing_optimizer():
    """Verify Simulated Annealing improves initial order cost."""
    initial = [0, 1, 2, 3, 4, 5]
    # Simple objective: sum of abs differences between consecutive elements
    def cost_fn(seq):
        return sum(abs(seq[i] - seq[i+1]) for i in range(len(seq) - 1))

    res = simulated_annealing_permutation(initial, cost_fn, max_iterations=50)
    assert res["final_cost"] <= res["initial_cost"]
