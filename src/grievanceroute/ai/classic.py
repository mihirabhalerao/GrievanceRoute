"""
Classic AI State-Space Demonstrations for GrievanceRoute.
Implements Water-Jug problem and 8-Puzzle with A* (Manhattan distance heuristic)
and mathematical solvability validation.
"""

from __future__ import annotations
import heapq
import math
from collections import deque
from typing import List, Tuple, Dict, Any, Optional


def solve_water_jug(
    cap1: int = 4,
    cap2: int = 3,
    target: int = 2,
    max_states: int = 1000,
) -> Dict[str, Any]:
    """
    Solve Water-Jug problem using Breadth-First Search.
    Mathematically verifies solvability: target <= max(cap1, cap2) and target % gcd(cap1, cap2) == 0.
    Returns optimal action sequence, state transitions, and visited states count.
    """
    # Solvability check
    g = math.gcd(cap1, cap2)
    if target > max(cap1, cap2) or target % g != 0:
        return {
            "solvable": False,
            "reason": f"Target {target} is not a multiple of gcd({cap1}, {cap2}) = {g} or exceeds max capacity.",
            "steps": [],
            "visited_count": 0,
        }

    queue: deque[Tuple[int, int, List[Tuple[str, int, int]]]] = deque([(0, 0, [("Initial State", 0, 0)])])
    visited = set([(0, 0)])

    while queue and len(visited) < max_states:
        j1, j2, history = queue.popleft()

        if j1 == target or j2 == target:
            return {
                "solvable": True,
                "target": target,
                "steps": history,
                "step_count": len(history) - 1,
                "visited_count": len(visited),
            }

        # Available transitions
        transitions = [
            ("Fill Jug 1", cap1, j2),
            ("Fill Jug 2", j1, cap2),
            ("Empty Jug 1", 0, j2),
            ("Empty Jug 2", j1, 0),
            # Pour Jug 1 -> Jug 2
            ("Pour Jug 1 -> Jug 2", max(0, j1 - (cap2 - j2)), min(cap2, j2 + j1)),
            # Pour Jug 2 -> Jug 1
            ("Pour Jug 2 -> Jug 1", min(cap1, j1 + j2), max(0, j2 - (cap1 - j1))),
        ]

        for action, next_j1, next_j2 in transitions:
            if (next_j1, next_j2) not in visited:
                visited.add((next_j1, next_j2))
                queue.append((next_j1, next_j2, history + [(action, next_j1, next_j2)]))

    return {
        "solvable": False,
        "reason": "Search limit reached without finding target state.",
        "steps": [],
        "visited_count": len(visited),
    }


# ---------------- 8-PUZZLE SOLVER ----------------

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)  # 0 denotes blank tile


def is_8_puzzle_solvable(state: Tuple[int, ...]) -> bool:
    """
    Check if an 8-puzzle board configuration is solvable using inversion count.
    In 8-puzzle (odd grid width 3), a state is solvable iff its inversion count is even.
    """
    tiles = [t for t in state if t != 0]
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions % 2 == 0


def manhattan_distance_8_puzzle(state: Tuple[int, ...]) -> int:
    """Calculate Manhattan distance lower bound heuristic to goal state."""
    distance = 0
    for idx, val in enumerate(state):
        if val == 0:
            continue
        # In goal state, val should be at index val - 1
        target_idx = val - 1
        curr_r, curr_c = divmod(idx, 3)
        goal_r, goal_c = divmod(target_idx, 3)
        distance += abs(curr_r - goal_r) + abs(curr_c - goal_c)
    return distance


def get_8_puzzle_neighbors(state: Tuple[int, ...]) -> List[Tuple[str, Tuple[int, ...]]]:
    """Generate legal sliding moves from current board configuration."""
    blank_idx = state.index(0)
    r, c = divmod(blank_idx, 3)
    moves = []

    directions = [("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1)]
    for name, dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_blank = nr * 3 + nc
            board = list(state)
            board[blank_idx], board[new_blank] = board[new_blank], board[blank_idx]
            moves.append((name, tuple(board)))
    return moves


def solve_8_puzzle(
    initial_state: Tuple[int, ...],
    algorithm: str = "A*",
    max_expanded: int = 5000,
) -> Dict[str, Any]:
    """
    Solve the 8-puzzle board using A* (f = g + h) or Greedy Best-First (f = h).
    Validates solvability upfront to prevent infinite search on impossible configurations.
    """
    if not is_8_puzzle_solvable(initial_state):
        return {
            "solvable": False,
            "reason": "Odd inversion count: configuration is mathematically impossible to reach standard goal.",
            "steps": [],
            "expanded_count": 0,
        }

    counter = 0
    h_start = manhattan_distance_8_puzzle(initial_state)
    f_start = h_start if algorithm == "A*" else h_start

    # open_set item: (priority, counter, g_score, state, path)
    open_set: List[Tuple[int, int, int, Tuple[int, ...], List[Tuple[str, Tuple[int, ...]]]]] = [
        (f_start, counter, 0, initial_state, [("START", initial_state)])
    ]
    visited = set()
    expanded_count = 0

    while open_set and expanded_count < max_expanded:
        f, _, g, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        expanded_count += 1

        if current == GOAL_STATE:
            return {
                "solvable": True,
                "algorithm": algorithm,
                "steps": path,
                "step_count": len(path) - 1,
                "expanded_count": expanded_count,
            }

        for move_name, neighbor in get_8_puzzle_neighbors(current):
            if neighbor not in visited:
                h = manhattan_distance_8_puzzle(neighbor)
                tentative_g = g + 1
                priority = tentative_g + h if algorithm == "A*" else h
                counter += 1
                heapq.heappush(open_set, (priority, counter, tentative_g, neighbor, path + [(move_name, neighbor)]))

    return {
        "solvable": False,
        "reason": f"Search limit ({max_expanded} states expanded) reached.",
        "steps": [],
        "expanded_count": expanded_count,
    }
