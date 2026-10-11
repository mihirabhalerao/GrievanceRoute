"""
AI Algorithm Lab Page for GrievanceRoute.
Interactive visualizers for A*, Greedy Best-First, Water-Jug, 8-Puzzle,
AO* AND-OR graphs, Minimax with Alpha-Beta pruning, and Simulated Annealing.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from grievanceroute.services.ui_theme import apply_custom_theme, render_hero
from grievanceroute.routing.road_graph import RoadGraphService
from grievanceroute.ai.search import a_star_search, greedy_best_first_search
from grievanceroute.ai.classic import solve_water_jug, solve_8_puzzle, is_8_puzzle_solvable
from grievanceroute.ai.game_and_or import solve_ao_star, minimax_alpha_beta


st.set_page_config(page_title="AI Algorithm Lab — GrievanceRoute", page_icon="🧩", layout="wide")
apply_custom_theme()

render_hero(
    title="AI Algorithm Laboratory",
    subtitle="Interactive heuristic search, state-space exploration, adversarial game trees, and classical AI benchmarks.",
    badge="Unit I, II & Lab Supplementary"
)

tab_search, tab_jug, tab_puzzle, tab_ao, tab_minimax = st.tabs([
    "🛣️ A* vs. Greedy Search",
    "🏺 Water-Jug Problem",
    "🧩 8-Puzzle Solver",
    "🌲 AO* (AND-OR Search)",
    "⚔️ Minimax & Alpha-Beta",
])

# 1. A* vs Greedy Road Search
with tab_search:
    st.subheader("Spatial Graph Search: A* vs. Greedy Best-First")
    st.caption("A* evaluates f(n) = g(n) + h(n) with admissible Euclidean distance lower bounds; Greedy Best-First considers only h(n).")

    road_svc = RoadGraphService()
    nodes = list(road_svc.node_coords.keys())

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        start_node = st.selectbox("Start Node:", nodes, index=0)
    with c_s2:
        goal_node = st.selectbox("Goal Node:", nodes, index=len(nodes) - 1)

    if st.button("🚀 Compare Search Algorithms", type="primary"):
        res_astar = a_star_search(road_svc.graph, start_node, goal_node, road_svc.node_coords)
        res_greedy = greedy_best_first_search(road_svc.graph, start_node, goal_node, road_svc.node_coords)

        col_a, col_g = st.columns(2)
        with col_a:
            st.markdown("#### ⭐ A* Algorithm (Admissible & Optimal)")
            st.metric("Total Path Cost", f"{res_astar['cost']} km")
            st.metric("Expanded Nodes", res_astar["expanded_count"])
            st.metric("Runtime", f"{res_astar['runtime_ms']} ms")
            st.write("**Path:**", " ➔ ".join(res_astar["path"]))

        with col_g:
            st.markdown("#### ⚡ Greedy Best-First (Heuristic Only)")
            st.metric("Total Path Cost", f"{res_greedy['cost']} km")
            st.metric("Expanded Nodes", res_greedy["expanded_count"])
            st.metric("Runtime", f"{res_greedy['runtime_ms']} ms")
            st.write("**Path:**", " ➔ ".join(res_greedy["path"]))

        if res_astar["cost"] < res_greedy["cost"]:
            st.success(f"A* found a strictly more optimal path ({res_astar['cost']} km vs {res_greedy['cost']} km), demonstrating admissibility guarantees.")
        else:
            st.info("Both algorithms reached the goal with identical path costs on this network query.")

# 2. Water-Jug Problem
with tab_jug:
    st.subheader("Water-Jug State-Space Problem (BFS)")
    st.caption("Solves for target water volume using Fill, Empty, and Pour state transitions. Solvable iff target ≤ max capacity and target % gcd(cap1, cap2) == 0.")

    j1, j2, jt = st.columns(3)
    with j1:
        cap1_val = st.number_input("Jug 1 Capacity (L):", min_value=1, max_value=20, value=4)
    with j2:
        cap2_val = st.number_input("Jug 2 Capacity (L):", min_value=1, max_value=20, value=3)
    with jt:
        target_val = st.number_input("Target Volume (L):", min_value=1, max_value=20, value=2)

    if st.button("Run Water-Jug Solver", type="primary"):
        jug_res = solve_water_jug(cap1_val, cap2_val, target_val)
        if not jug_res["solvable"]:
            st.error(f"Unsolvable: {jug_res['reason']}")
        else:
            st.success(f"Target of {target_val}L reached in **{jug_res['step_count']} steps** (States explored: {jug_res['visited_count']})!")
            steps_df = pd.DataFrame(
                [{"Step": i, "Action Taken": a, "Jug 1 (L)": j1, "Jug 2 (L)": j2} for i, (a, j1, j2) in enumerate(jug_res["steps"])]
            )
            st.dataframe(steps_df, use_container_width=True)

# 3. 8-Puzzle Solver
with tab_puzzle:
    st.subheader("8-Puzzle State-Space Search (A* with Manhattan Distance)")
    st.caption("Goal State: [1, 2, 3, 4, 5, 6, 7, 8, 0 (Blank)]. Odd inversion count denotes mathematically unsolvable configurations.")

    preset = st.selectbox("Choose Initial Configuration:", [
        "Easy (2 moves to goal)",
        "Medium (6 moves to goal)",
        "Unsolvable Test State (Inversion count check)"
    ])

    presets_map = {
        "Easy (2 moves to goal)": (1, 2, 3, 4, 5, 6, 7, 0, 8),
        "Medium (6 moves to goal)": (1, 2, 3, 0, 4, 6, 7, 5, 8),
        "Unsolvable Test State (Inversion count check)": (1, 2, 3, 4, 5, 6, 8, 7, 0),
    }

    initial_board = presets_map[preset]
    is_solv = is_8_puzzle_solvable(initial_board)

    col_b, col_stat = st.columns([1, 2])
    with col_b:
        st.markdown("**Initial Board:**")
        b_matrix = [initial_board[i:i+3] for i in range(0, 9, 3)]
        st.write(pd.DataFrame(b_matrix))
    with col_stat:
        st.write(f"Solvable by inversion parity: **{'Yes' if is_solv else 'No (Impossible)'}**")

    if st.button("Solve 8-Puzzle with A*", type="primary"):
        puz_res = solve_8_puzzle(initial_board, algorithm="A*")
        if not puz_res["solvable"]:
            st.error(f"Failed: {puz_res['reason']}")
        else:
            st.success(f"Goal solved in **{puz_res['step_count']} steps** (States expanded: {puz_res['expanded_count']})!")
            for idx, (move, board) in enumerate(puz_res["steps"]):
                with st.expander(f"Step {idx}: Action [{move}]"):
                    st.write(pd.DataFrame([board[k:k+3] for k in range(0, 9, 3)]))

# 4. AO* (AND-OR Graph Search)
with tab_ao:
    st.subheader("AO* Search on AND-OR Graphs")
    st.caption("Demonstrates bottom-up cost updates and marked solution subtree extraction for problem decomposition.")

    SAMPLE_AND_OR_GRAPH = {
        "A": {"h": 10.0, "branches": [
            {"type": "OR", "children": ["B"], "edge_cost": 1.0},
            {"type": "AND", "children": ["C", "D"], "edge_cost": 2.0},
        ]},
        "B": {"h": 6.0, "branches": [
            {"type": "OR", "children": ["G"], "edge_cost": 1.0},
            {"type": "OR", "children": ["H"], "edge_cost": 1.0},
        ]},
        "C": {"h": 2.0, "branches": []},  # terminal solved
        "D": {"h": 3.0, "branches": []},  # terminal solved
        "G": {"h": 0.0, "branches": []},  # terminal solved
        "H": {"h": 0.0, "branches": []},  # terminal solved
    }

    st.json(SAMPLE_AND_OR_GRAPH)
    if st.button("Execute AO* Algorithm", type="primary"):
        ao_res = solve_ao_star(SAMPLE_AND_OR_GRAPH, root_name="A")
        st.success(f"AO* Terminated! Solved: {ao_res['solved']}, Total Cost: {ao_res['total_cost']}, Iterations: {ao_res['iterations']}")
        st.json(ao_res["solution_tree"])

# 5. Minimax & Alpha-Beta
with tab_minimax:
    st.subheader("Adversarial Minimax Search with Alpha-Beta Pruning")
    st.caption("Evaluates a 3-ply game tree and demonstrates branch pruning.")

    GAME_TREE = {
        "root": {"children": ["n1", "n2"]},
        "n1": {"children": ["n11", "n12"]},
        "n2": {"children": ["n21", "n22"]},
        "n11": {"value": 3.0},
        "n12": {"value": 5.0},
        "n21": {"value": 6.0},
        "n22": {"value": 9.0},
    }

    st.write("**Game Tree Structure:** Root (MAX) -> Level 1 (MIN) -> Leaves (Terminal Values: 3, 5, 6, 9)")
    if st.button("Execute Alpha-Beta Pruning", type="primary"):
        stats = {"leaves_evaluated": 0, "pruned_branches": 0}
        val, best_move = minimax_alpha_beta(GAME_TREE, "root", depth=2, alpha=float("-inf"), beta=float("inf"), is_maximizing=True, stats=stats)
        st.success(f"Optimal Minimax Value: **{val}** (Best Move: `{best_move}`)")
        st.metric("Leaves Evaluated", stats["leaves_evaluated"])
        st.metric("Pruned Subtrees", stats["pruned_branches"])
