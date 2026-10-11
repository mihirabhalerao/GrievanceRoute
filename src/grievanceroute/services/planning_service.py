"""
Planning Service for GrievanceRoute.
Orchestrates crew assignments, shift evaluation, route solvers, and benchmark comparisons.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from grievanceroute.routing.vrp_solver import (
    solve_greedy_vrp,
    solve_hill_climbing_vrp,
    solve_simulated_annealing_vrp,
    compare_vrp_solvers,
)
from grievanceroute.routing.road_graph import RoadGraphService


class PlanningService:
    """Manages crew dispatch, route optimization, and schedule generation."""

    def __init__(
        self,
        crews_csv: str | Path = "data/raw/crews.csv",
        road_network_json: str | Path = "data/raw/road_network.json",
    ):
        self.crews_csv = Path(crews_csv)
        self.road_network_json = Path(road_network_json)
        self.road_graph_svc = RoadGraphService(self.road_network_json)

    def load_crews(self) -> pd.DataFrame:
        """Load configured repair crews."""
        if self.crews_csv.exists():
            return pd.read_csv(self.crews_csv)
        return pd.DataFrame()

    def generate_plan(
        self,
        complaints: List[Dict[str, Any]],
        algorithm: str = "Simulated Annealing",
        alpha: float = 1.0,
        beta: float = 10.0,
        gamma: float = 5.0,
    ) -> Dict[str, Any]:
        """Generate dispatch schedule using selected solver."""
        crews_df = self.load_crews()
        if crews_df.empty:
            return {"error": "No crew records found."}

        greedy_res = solve_greedy_vrp(complaints, crews_df, alpha, beta, gamma)

        if algorithm == "Greedy Baseline":
            return greedy_res
        elif algorithm == "Hill Climbing":
            unserved = [c for c in complaints if c.get("complaint_id") not in [
                s for r in greedy_res["crew_reports"] for s in r["stops"]
            ]]
            return solve_hill_climbing_vrp(greedy_res["schedules"], crews_df, unserved, alpha, beta, gamma)
        else:  # Simulated Annealing
            unserved = [c for c in complaints if c.get("complaint_id") not in [
                s for r in greedy_res["crew_reports"] for s in r["stops"]
            ]]
            return solve_simulated_annealing_vrp(greedy_res["schedules"], crews_df, unserved, alpha, beta, gamma)

    def compare_all_solvers(
        self,
        complaints: List[Dict[str, Any]],
        alpha: float = 1.0,
        beta: float = 10.0,
        gamma: float = 5.0,
    ) -> pd.DataFrame:
        """Run all solvers on complaints and return comparison summary DataFrame."""
        crews_df = self.load_crews()
        return compare_vrp_solvers(complaints, crews_df, alpha, beta, gamma)
