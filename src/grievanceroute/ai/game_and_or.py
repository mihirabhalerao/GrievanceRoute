"""
AO* (AND-OR Graph Search) and Minimax with Alpha-Beta Pruning for GrievanceRoute.
Demonstrates heuristic decomposition and adversarial search.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional


class ANDORNode:
    """Node in an AND-OR graph."""
    def __init__(self, name: str, heuristic: float = 0.0):
        self.name = name
        self.heuristic = heuristic
        self.solved = False
        # branches: list of dicts: {"type": "OR"|"AND", "children": [node_names], "edge_cost": float}
        self.branches: List[Dict[str, Any]] = []
        self.best_branch: Optional[Dict[str, Any]] = None


def solve_ao_star(
    graph_dict: Dict[str, Dict[str, Any]],
    root_name: str = "A",
    max_cycles: int = 50,
) -> Dict[str, Any]:
    """
    AO* search over an AND-OR graph representation.
    Iteratively expands non-terminal leaf nodes, updates heuristic costs bottom-up,
    and marks optimal solution arcs.
    """
    # Build graph data structure
    nodes: Dict[str, ANDORNode] = {}
    for name, data in graph_dict.items():
        node = ANDORNode(name, heuristic=data.get("h", 0.0))
        node.branches = data.get("branches", [])
        if not node.branches:
            node.solved = True  # Terminal leaf node
        nodes[name] = node

    def compute_branch_cost(branch: Dict[str, Any]) -> float:
        edge_cost = branch.get("edge_cost", 1.0)
        children = branch.get("children", [])
        child_costs = sum(nodes[c].heuristic for c in children)
        return edge_cost + child_costs

    cycle = 0
    while not nodes[root_name].solved and cycle < max_cycles:
        cycle += 1
        # Traverse marked best path from root to find an unsolved node
        curr = root_name
        path_to_expand = [curr]
        while nodes[curr].branches and not nodes[curr].solved:
            if nodes[curr].best_branch is None:
                break
            # Pick first unsolved child in best branch
            unsolved_child = None
            for c in nodes[curr].best_branch["children"]:
                if not nodes[c].solved:
                    unsolved_child = c
                    break
            if unsolved_child is None:
                nodes[curr].solved = True
                break
            curr = unsolved_child
            path_to_expand.append(curr)

        # Bottom-up cost update along path
        for n_name in reversed(path_to_expand):
            node = nodes[n_name]
            if not node.branches:
                continue

            best_cost = float("inf")
            best_br = None
            all_children_solved = False

            for br in node.branches:
                b_cost = compute_branch_cost(br)
                if b_cost < best_cost:
                    best_cost = b_cost
                    best_br = br
                    all_children_solved = all(nodes[c].solved for c in br["children"])

            node.heuristic = best_cost
            node.best_branch = best_br
            node.solved = all_children_solved

    # Extract solution tree from root
    def extract_solution_tree(name: str) -> Dict[str, Any]:
        node = nodes[name]
        tree = {"node": name, "cost": node.heuristic, "solved": node.solved}
        if node.best_branch:
            tree["type"] = node.best_branch["type"]
            tree["children"] = [extract_solution_tree(c) for c in node.best_branch["children"]]
        return tree

    return {
        "root": root_name,
        "solved": nodes[root_name].solved,
        "total_cost": round(nodes[root_name].heuristic, 2),
        "iterations": cycle,
        "solution_tree": extract_solution_tree(root_name),
    }


def minimax_alpha_beta(
    tree: Dict[str, Any],
    node_id: str,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    stats: Dict[str, int],
) -> Tuple[float, Optional[str]]:
    """
    Adversarial Minimax Search with Alpha-Beta pruning.
    Returns (optimal_value, best_action_child).
    Updates evaluation and pruning counters in stats dict.
    """
    node = tree[node_id]
    children = node.get("children", [])

    # Terminal node
    if not children or depth == 0:
        stats["leaves_evaluated"] += 1
        return float(node.get("value", 0.0)), None

    best_action = None

    if is_maximizing:
        max_eval = float("-inf")
        for child_id in children:
            eval_val, _ = minimax_alpha_beta(tree, child_id, depth - 1, alpha, beta, False, stats)
            if eval_val > max_eval:
                max_eval = eval_val
                best_action = child_id
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                stats["pruned_branches"] += 1
                break  # Beta cut-off
        return max_eval, best_action
    else:
        min_eval = float("inf")
        for child_id in children:
            eval_val, _ = minimax_alpha_beta(tree, child_id, depth - 1, alpha, beta, True, stats)
            if eval_val < min_eval:
                min_eval = eval_val
                best_action = child_id
            beta = min(beta, eval_val)
            if beta <= alpha:
                stats["pruned_branches"] += 1
                break  # Alpha cut-off
        return min_eval, best_action
