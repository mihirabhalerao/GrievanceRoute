"""
Local Search and Heuristic Optimization for GrievanceRoute.
Implements Hill Climbing and Simulated Annealing for combinatorial sequence optimization.
"""

from __future__ import annotations
import math
import random
import time
from typing import List, Callable, Dict, Any, Tuple


def hill_climbing_permutation(
    initial_solution: List[Any],
    cost_fn: Callable[[List[Any]], float],
    max_evaluations: int = 500,
) -> Dict[str, Any]:
    """
    Steepest-descent Hill Climbing over permutation space using 2-opt pairwise swaps.
    Terminates when no neighbor improves the objective or max evaluations reached.
    """
    start_time = time.time()
    current_solution = list(initial_solution)
    current_cost = cost_fn(current_solution)
    history: List[float] = [round(current_cost, 3)]
    eval_count = 1

    improved = True
    while improved and eval_count < max_evaluations:
        improved = False
        best_neighbor = None
        best_neighbor_cost = current_cost

        # Generate 2-opt neighbors
        n = len(current_solution)
        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbor = list(current_solution)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                eval_count += 1
                n_cost = cost_fn(neighbor)

                if n_cost < best_neighbor_cost:
                    best_neighbor_cost = n_cost
                    best_neighbor = neighbor
                    improved = True

                if eval_count >= max_evaluations:
                    break
            if eval_count >= max_evaluations:
                break

        if improved and best_neighbor is not None:
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            history.append(round(current_cost, 3))

    duration_ms = (time.time() - start_time) * 1000
    return {
        "initial_cost": history[0],
        "final_cost": round(current_cost, 3),
        "best_solution": current_solution,
        "evaluations": eval_count,
        "cost_history": history,
        "runtime_ms": round(duration_ms, 3),
    }


def simulated_annealing_permutation(
    initial_solution: List[Any],
    cost_fn: Callable[[List[Any]], float],
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    min_temp: float = 0.1,
    max_iterations: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulated Annealing over sequence permutations.
    Accepts inferior moves with Boltzmann probability P = exp(-delta_E / T) to escape local minima.
    Tracks best-so-far solution independently.
    """
    py_rand = random.Random(seed)
    start_time = time.time()

    current_solution = list(initial_solution)
    current_cost = cost_fn(current_solution)

    best_solution = list(current_solution)
    best_cost = current_cost

    temperature = initial_temp
    history: List[float] = [round(current_cost, 3)]
    best_history: List[float] = [round(best_cost, 3)]

    iteration = 0
    while temperature > min_temp and iteration < max_iterations:
        iteration += 1
        n = len(current_solution)
        if n < 2:
            break

        # Random 2-opt swap move
        i, j = py_rand.sample(range(n), 2)
        neighbor = list(current_solution)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_cost = cost_fn(neighbor)

        delta = neighbor_cost - current_cost

        # Metropolis criterion
        if delta < 0 or (temperature > 1e-6 and py_rand.random() < math.exp(-delta / temperature)):
            current_solution = neighbor
            current_cost = neighbor_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = list(current_solution)

        history.append(round(current_cost, 3))
        best_history.append(round(best_cost, 3))
        temperature *= cooling_rate

    duration_ms = (time.time() - start_time) * 1000
    return {
        "initial_cost": history[0],
        "final_cost": round(best_cost, 3),
        "best_solution": best_solution,
        "iterations": iteration,
        "cost_history": history,
        "best_cost_history": best_history,
        "runtime_ms": round(duration_ms, 3),
    }
