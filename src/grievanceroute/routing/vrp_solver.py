"""
Multi-Crew Vehicle Routing Problem (VRP) Solver for GrievanceRoute.
Implements:
1. Greedy Nearest-High-Priority Baseline
2. Hill Climbing Route Improvement
3. Simulated Annealing Route Optimizer
Evaluates objective: J(R) = alpha*Distance + beta*MissedUrgency + gamma*Overtime.
"""

from __future__ import annotations
import math
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

from grievanceroute.ai.search import euclidean_distance_km
from grievanceroute.ai.local_search import hill_climbing_permutation, simulated_annealing_permutation
from grievanceroute.routing.crew_constraints import check_skill_compatibility, evaluate_crew_shift


def calculate_route_distance(
    depot_lat: float,
    depot_lon: float,
    complaint_sequence: List[Dict[str, Any]],
) -> float:
    """Calculate total closed tour distance (depot -> complaints -> depot) in km."""
    if not complaint_sequence:
        return 0.0

    total_dist = 0.0
    curr_lat, curr_lon = depot_lat, depot_lon

    for c in complaint_sequence:
        c_lat = float(c["latitude"])
        c_lon = float(c["longitude"])
        total_dist += euclidean_distance_km(curr_lat, curr_lon, c_lat, c_lon)
        curr_lat, curr_lon = c_lat, c_lon

    # Return to depot
    total_dist += euclidean_distance_km(curr_lat, curr_lon, depot_lat, depot_lon)
    return total_dist


def evaluate_vrp_objective(
    crew_schedules: Dict[str, List[Dict[str, Any]]],
    crews_df: pd.DataFrame,
    unassigned_complaints: List[Dict[str, Any]],
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate global objective:
    J(R) = alpha * total_dist + beta * unserved_urgency + gamma * total_overtime
    """
    total_distance = 0.0
    total_overtime = 0.0
    served_count = 0
    urgency_served = 0.0
    crew_reports = []

    for _, crew in crews_df.iterrows():
        cid = crew["crew_id"]
        seq = crew_schedules.get(cid, [])
        d_lat = float(crew["depot_lat"])
        d_lon = float(crew["depot_lon"])
        speed = float(crew.get("speed_kmh", 25.0))
        shift_limit = float(crew.get("shift_hours", 8.0))

        dist = calculate_route_distance(d_lat, d_lon, seq)
        shift_eval = evaluate_crew_shift(crew.to_dict(), seq, dist, speed_kmh=speed, max_shift_hours=shift_limit)

        total_distance += dist
        total_overtime += shift_eval["overtime_hours"]
        served_count += len(seq)
        urgency_served += sum(float(c.get("urgency_score", 3)) for c in seq)

        crew_reports.append({
            "crew_id": cid,
            "crew_name": crew["crew_name"],
            "skill": crew["skill"],
            "jobs_count": len(seq),
            "distance_km": round(dist, 2),
            "travel_hours": shift_eval["travel_hours"],
            "service_hours": shift_eval["service_hours"],
            "total_hours": shift_eval["total_hours"],
            "overtime_hours": shift_eval["overtime_hours"],
            "is_feasible": shift_eval["is_feasible"],
            "violations": shift_eval["violations"],
            "stops": [c.get("complaint_id") for c in seq],
        })

    unserved_urgency = sum(float(c.get("urgency_score", 3)) for c in unassigned_complaints)
    total_cost = alpha * total_distance + beta * unserved_urgency + gamma * total_overtime

    details = {
        "total_cost": round(total_cost, 2),
        "total_distance_km": round(total_distance, 2),
        "unserved_urgency_penalty": round(unserved_urgency, 2),
        "total_overtime_hours": round(total_overtime, 2),
        "served_count": served_count,
        "unserved_count": len(unassigned_complaints),
        "urgency_served": round(urgency_served, 2),
        "crew_reports": crew_reports,
    }
    return total_cost, details


def solve_greedy_vrp(
    complaints: List[Dict[str, Any]],
    crews_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
) -> Dict[str, Any]:
    """
    Greedy Nearest-High-Priority Assignment:
    Sorts complaints by urgency descending, then assigns to compatible nearest available crew.
    """
    start_time = time.time()
    crew_schedules: Dict[str, List[Dict[str, Any]]] = {c["crew_id"]: [] for _, c in crews_df.iterrows()}
    crew_capacities = {c["crew_id"]: int(c["capacity"]) for _, c in crews_df.iterrows()}
    crew_skills = {c["crew_id"]: c["skill"] for _, c in crews_df.iterrows()}

    # Sort candidates by urgency descending
    sorted_complaints = sorted(complaints, key=lambda c: float(c.get("urgency_score", 3)), reverse=True)
    unassigned: List[Dict[str, Any]] = []

    for comp in sorted_complaints:
        assigned = False
        c_lat, c_lon = float(comp["latitude"]), float(comp["longitude"])
        c_type = comp["complaint_type"]

        # Find compatible crews that have remaining capacity
        compatible_crews = [
            cid for cid, skill in crew_skills.items()
            if check_skill_compatibility(skill, c_type) and len(crew_schedules[cid]) < crew_capacities[cid]
        ]

        if compatible_crews:
            # Pick crew whose current last position or depot is closest
            best_cid = None
            best_dist = float("inf")
            for cid in compatible_crews:
                c_row = crews_df[crews_df["crew_id"] == cid].iloc[0]
                if crew_schedules[cid]:
                    last_c = crew_schedules[cid][-1]
                    dist = euclidean_distance_km(last_c["latitude"], last_c["longitude"], c_lat, c_lon)
                else:
                    dist = euclidean_distance_km(c_row["depot_lat"], c_row["depot_lon"], c_lat, c_lon)

                if dist < best_dist:
                    best_dist = dist
                    best_cid = cid

            if best_cid:
                crew_schedules[best_cid].append(comp)
                assigned = True

        if not assigned:
            unassigned.append(comp)

    cost, details = evaluate_vrp_objective(crew_schedules, crews_df, unassigned, alpha, beta, gamma)
    details["algorithm"] = "Greedy (Nearest-High-Priority)"
    details["runtime_ms"] = round((time.time() - start_time) * 1000, 2)
    details["schedules"] = crew_schedules
    return details


def solve_hill_climbing_vrp(
    initial_schedule: Dict[str, List[Dict[str, Any]]],
    crews_df: pd.DataFrame,
    unassigned_complaints: List[Dict[str, Any]],
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
) -> Dict[str, Any]:
    """Improve stop sequences for each crew using 2-opt Hill Climbing."""
    start_time = time.time()
    optimized_schedules: Dict[str, List[Dict[str, Any]]] = {}

    for _, crew in crews_df.iterrows():
        cid = crew["crew_id"]
        seq = initial_schedule.get(cid, [])
        if len(seq) > 2:
            d_lat = float(crew["depot_lat"])
            d_lon = float(crew["depot_lon"])
            def dist_fn(p_seq: List[Dict[str, Any]]) -> float:
                return calculate_route_distance(d_lat, d_lon, p_seq)

            hc_res = hill_climbing_permutation(seq, dist_fn, max_evaluations=300)
            optimized_schedules[cid] = hc_res["best_solution"]
        else:
            optimized_schedules[cid] = list(seq)

    cost, details = evaluate_vrp_objective(optimized_schedules, crews_df, unassigned_complaints, alpha, beta, gamma)
    details["algorithm"] = "A* / 2-Opt Hill Climbing"
    details["runtime_ms"] = round((time.time() - start_time) * 1000, 2)
    details["schedules"] = optimized_schedules
    return details


def solve_simulated_annealing_vrp(
    initial_schedule: Dict[str, List[Dict[str, Any]]],
    crews_df: pd.DataFrame,
    unassigned_complaints: List[Dict[str, Any]],
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Improve stop sequences for each crew using Simulated Annealing."""
    start_time = time.time()
    optimized_schedules: Dict[str, List[Dict[str, Any]]] = {}

    for _, crew in crews_df.iterrows():
        cid = crew["crew_id"]
        seq = initial_schedule.get(cid, [])
        if len(seq) > 2:
            d_lat = float(crew["depot_lat"])
            d_lon = float(crew["depot_lon"])
            def dist_fn(p_seq: List[Dict[str, Any]]) -> float:
                return calculate_route_distance(d_lat, d_lon, p_seq)

            sa_res = simulated_annealing_permutation(seq, dist_fn, initial_temp=80.0, cooling_rate=0.92, max_iterations=250, seed=seed)
            optimized_schedules[cid] = sa_res["best_solution"]
        else:
            optimized_schedules[cid] = list(seq)

    cost, details = evaluate_vrp_objective(optimized_schedules, crews_df, unassigned_complaints, alpha, beta, gamma)
    details["algorithm"] = "Simulated Annealing Optimizer"
    details["runtime_ms"] = round((time.time() - start_time) * 1000, 2)
    details["schedules"] = optimized_schedules
    return details


def compare_vrp_solvers(
    complaints: List[Dict[str, Any]],
    crews_df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 10.0,
    gamma: float = 5.0,
) -> pd.DataFrame:
    """Run all 3 routing methods on identical complaints and return a comparative DataFrame."""
    res_greedy = solve_greedy_vrp(complaints, crews_df, alpha, beta, gamma)
    res_hc = solve_hill_climbing_vrp(res_greedy["schedules"], crews_df, complaints[res_greedy["served_count"]:], alpha, beta, gamma)
    res_sa = solve_simulated_annealing_vrp(res_greedy["schedules"], crews_df, complaints[res_greedy["served_count"]:], alpha, beta, gamma)

    records = []
    base_dist = res_greedy["total_distance_km"]

    for r in [res_greedy, res_hc, res_sa]:
        dist = r["total_distance_km"]
        dist_improv = round(((base_dist - dist) / base_dist * 100), 2) if base_dist > 0 else 0.0
        records.append({
            "Algorithm": r["algorithm"],
            "Total Distance (km)": dist,
            "Distance Improvement (%)": dist_improv,
            "Objective Cost J(R)": r["total_cost"],
            "Complaints Served": r["served_count"],
            "Urgency Served": r["urgency_served"],
            "Unserved Work": r["unserved_count"],
            "Total Overtime (h)": r["total_overtime_hours"],
            "Runtime (ms)": r["runtime_ms"],
        })

    return pd.DataFrame(records)
