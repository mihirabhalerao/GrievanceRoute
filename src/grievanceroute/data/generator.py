"""
Synthetic Dataset Generator for GrievanceRoute.
Generates reproducible Pune civic complaints, repair crews, and road-network graph.
"""

from __future__ import annotations
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


PUNE_ZONES: Dict[str, Dict[str, Any]] = {
    "Kothrud": {"lat": 18.5074, "lon": 73.8077, "low_lying": False},
    "Shivajinagar": {"lat": 18.5314, "lon": 73.8446, "low_lying": True},
    "Hadapsar": {"lat": 18.5089, "lon": 73.9259, "low_lying": True},
    "Baner": {"lat": 18.5590, "lon": 73.7868, "low_lying": False},
    "Swargate": {"lat": 18.5018, "lon": 73.8586, "low_lying": True},
    "Viman Nagar": {"lat": 18.5679, "lon": 73.9143, "low_lying": False},
}

CATEGORY_CREW_MAP = {
    "pothole": "Road Maintenance",
    "water leak": "Water Supply Crew",
    "streetlight": "Electrical Crew",
    "garbage": "Sanitation Crew",
}

TEXT_TEMPLATES = {
    "pothole": [
        "Deep dangerous pothole on the main road near {loc}, causing heavy bike accidents.",
        "Sadak par bahut bada gaddha hai near {loc}, vehicles are getting damaged daily.",
        "Road surface broken with multiple crater-like potholes near {loc} chowk.",
        "Dangerous open trench and deep crater on the street opposite {loc} market.",
        "Severe road damage and uneven asphalt causing water accumulation near {loc}.",
        "Gaddhe pad gaye hain road pe at {loc}, two wheelers slipping regularly.",
        "Large pothole in the middle of the bus lane near {loc} signal.",
        "Big crater formed after rains near {loc}. Immediate tarring repair required.",
    ],
    "water leak": [
        "Main pipeline burst and clean drinking water is gushing onto the road at {loc}.",
        "Paani ki pipeline leak ho gayi hai near {loc}, lakhs of liters wasted since morning.",
        "Severe water logging due to underground drinking water pipe rupture near {loc} garden.",
        "Low water pressure and dirty contaminated tap supply reported across {loc} colony.",
        "Water pipe cracked near the storm drain at {loc}, road completely flooded.",
        "Nal se ganda paani aa raha hai and pipeline leaking continuously at {loc}.",
        "Major municipal valve leakage flooding residential baseline near {loc} society.",
        "Continuous water leakage from valve box at {loc} corner.",
    ],
    "streetlight": [
        "Dark street! All streetlights are completely off and dark near {loc} lane.",
        "Live electrical wire hanging dangerously from a bent light pole near {loc}.",
        "Streetlight pole sparking continuously near {loc} bus stand, risk of electrocution.",
        "Bijli ka pole band hai aur raat ko bahut andhera rehta hai near {loc}.",
        "Exposed underground wires near the streetlight junction box at {loc}.",
        "Streetlight flickering violently and short-circuit sparks visible at {loc}.",
        "Complete blackout on the municipal road stretch near {loc} school.",
        "Streetlights have not been functioning for the past 5 days at {loc} bypass.",
    ],
    "garbage": [
        "Huge pile of uncollected garbage rotting and stinking on the corner of {loc}.",
        "Kachra petti overflow ho rahi hai and stray animals spreading it near {loc}.",
        "Solid waste dumped on the footpath blocking pedestrians near {loc} market.",
        "Sanitation vehicle has not visited for four days; waste piling up at {loc}.",
        "Open dumping and burning of plastic garbage creating hazardous smoke at {loc}.",
        "Dead animal and medical waste unattended near public bin at {loc} colony.",
        "Overflowing dustbin with foul smell attracting mosquitoes near {loc} hospital.",
        "Garbage dumping spot created illegally near the drainage line at {loc}.",
    ],
}

LOCALITY_NAMES = [
    "JM Road", "FC Road", "Karve Road", "Paud Road", "MG Road",
    "Magarpatta Junction", "DP Road", "University Circle", "Balgandharva Chowk",
    "Shaniwar Wada", "Deccan Gymkhana", "Pashan Circle", "Aundh Link Road",
    "Kalyani Nagar Bridge", "Airport Road", "Phoenix Mall Corner", "Katraj Chowk",
]


def generate_complaint_record(
    index: int,
    rng: np.random.Generator,
    py_rand: random.Random,
    add_defects: bool = False
) -> Dict[str, Any]:
    """Generate a single realistic complaint record."""
    category = py_rand.choice(list(CATEGORY_CREW_MAP.keys()))
    zone = py_rand.choice(list(PUNE_ZONES.keys()))
    zone_info = PUNE_ZONES[zone]

    # Sample coordinates around zone center
    lat = float(rng.normal(loc=zone_info["lat"], scale=0.012))
    lon = float(rng.normal(loc=zone_info["lon"], scale=0.012))

    # Time within past 120 days
    days_ago = py_rand.randint(1, 120)
    hour_probs = np.array([
        0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 00-05
        0.05, 0.08, 0.09, 0.08, 0.07, 0.06,  # 06-11
        0.05, 0.05, 0.05, 0.06, 0.07, 0.08,  # 12-17
        0.07, 0.05, 0.03, 0.02, 0.01, 0.01   # 18-23
    ])
    hour_probs = hour_probs / hour_probs.sum()
    hour = int(rng.choice(range(24), p=hour_probs))
    minute = py_rand.randint(0, 59)
    base_date = datetime(2026, 9, 15) - timedelta(days=days_ago)
    complaint_time = base_date.replace(hour=hour, minute=minute, second=0)
    day_of_week = complaint_time.weekday()

    # Weather
    is_monsoon_month = complaint_time.month in [6, 7, 8, 9]
    monsoon = bool(is_monsoon_month)
    if is_monsoon_month:
        weather = py_rand.choices(["Rainy", "Stormy", "Cloudy", "Clear"], weights=[0.55, 0.20, 0.15, 0.10])[0]
    else:
        weather = py_rand.choices(["Clear", "Cloudy", "Rainy"], weights=[0.75, 0.20, 0.05])[0]

    low_lying = zone_info["low_lying"]
    repeat = bool(py_rand.random() < 0.18)

    # Historical resolution hours context
    hist_base = {
        "pothole": 36.0,
        "water leak": 18.0,
        "streetlight": 24.0,
        "garbage": 12.0
    }[category]
    historical_resolution_hours = round(float(rng.normal(loc=hist_base, scale=4.0)), 1)
    historical_resolution_hours = max(4.0, historical_resolution_hours)

    # Text description
    loc_name = py_rand.choice(LOCALITY_NAMES)
    template = py_rand.choice(TEXT_TEMPLATES[category])
    complaint_text = template.format(loc=loc_name)

    # Calculate urgency score (1-5) based on interacting factors
    urgency_raw = 2.0
    if category == "water leak":
        urgency_raw += 0.8
        if monsoon or low_lying:
            urgency_raw += 1.0
    elif category == "streetlight":
        if "wire" in complaint_text.lower() or "spark" in complaint_text.lower():
            urgency_raw += 2.2
        elif hour >= 19 or hour <= 5:
            urgency_raw += 0.5
    elif category == "pothole":
        if "accident" in complaint_text.lower() or "crater" in complaint_text.lower():
            urgency_raw += 1.2
        if monsoon:
            urgency_raw += 0.8
    elif category == "garbage":
        if "hospital" in complaint_text.lower() or "dead animal" in complaint_text.lower():
            urgency_raw += 1.8
        if repeat:
            urgency_raw += 0.6

    if repeat:
        urgency_raw += 0.7
    if weather == "Stormy":
        urgency_raw += 0.6

    # Add noise to urgency
    urgency_raw += float(rng.normal(0, 0.35))
    urgency_score = int(np.clip(round(urgency_raw), 1, 5))

    # Actual resolution hours (target for regression)
    # Higher urgency gets faster attention, weather causes delays
    delay_factor = 1.0
    if weather in ["Rainy", "Stormy"]:
        delay_factor += 0.4
    if repeat:
        delay_factor += 0.2

    base_time = {
        1: 52.0,
        2: 38.0,
        3: 26.0,
        4: 16.0,
        5: 8.0
    }[urgency_score]

    actual_hours = base_time * delay_factor + float(rng.exponential(scale=6.0))
    actual_resolution_hours = round(float(np.clip(actual_hours, 1.5, 168.0)), 1)

    # Status
    status_choices = ["Resolved", "In Progress", "Assigned", "Pending"]
    status_weights = [0.65, 0.15, 0.12, 0.08]
    status = py_rand.choices(status_choices, weights=status_weights)[0]

    record = {
        "complaint_id": f"PUN-GRV-{index+1:05d}",
        "complaint_text": complaint_text,
        "complaint_type": category,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "zone": zone,
        "timestamp": complaint_time.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": hour,
        "day_of_week": day_of_week,
        "weather": weather,
        "monsoon": monsoon,
        "low_lying_area": low_lying,
        "repeat_complaint": repeat,
        "historical_resolution_hours": historical_resolution_hours,
        "urgency_score": urgency_score,
        "actual_resolution_hours": actual_resolution_hours,
        "status": status,
        "crew_required": CATEGORY_CREW_MAP[category],
    }

    # Introduce subtle controlled defects if requested (for preprocessing lab exercises)
    if add_defects:
        defect_prob = py_rand.random()
        if defect_prob < 0.015:
            # Null weather
            record["weather"] = None
        elif defect_prob < 0.025:
            # Leading/trailing whitespace
            record["complaint_text"] = "   " + record["complaint_text"] + "  "
        elif defect_prob < 0.035:
            # Inconsistent casing in zone
            record["zone"] = record["zone"].lower()

    return record


def generate_crews() -> pd.DataFrame:
    """Generate crews configuration dataframe."""
    crews_data = [
        {"crew_id": "CRW-RD-01", "crew_name": "Kothrud Road Works", "depot_node": "DEPOT-KOT", "depot_lat": 18.5074, "depot_lon": 73.8077, "skill": "Road Maintenance", "capacity": 7, "shift_hours": 8.0, "speed_kmh": 25.0},
        {"crew_id": "CRW-RD-02", "crew_name": "Hadapsar Asphalt Team", "depot_node": "DEPOT-HAD", "depot_lat": 18.5089, "depot_lon": 73.9259, "skill": "Road Maintenance", "capacity": 6, "shift_hours": 8.0, "speed_kmh": 25.0},
        {"crew_id": "CRW-WS-01", "crew_name": "Shivajinagar Hydro Squad", "depot_node": "DEPOT-SHI", "depot_lat": 18.5314, "depot_lon": 73.8446, "skill": "Water Supply Crew", "capacity": 6, "shift_hours": 8.0, "speed_kmh": 22.0},
        {"crew_id": "CRW-WS-02", "crew_name": "Swargate Pipeline Quick Response", "depot_node": "DEPOT-SWA", "depot_lat": 18.5018, "depot_lon": 73.8586, "skill": "Water Supply Crew", "capacity": 5, "shift_hours": 8.0, "speed_kmh": 22.0},
        {"crew_id": "CRW-EL-01", "crew_name": "Baner Power & Lighting", "depot_node": "DEPOT-BAN", "depot_lat": 18.5590, "depot_lon": 73.7868, "skill": "Electrical Crew", "capacity": 8, "shift_hours": 8.0, "speed_kmh": 28.0},
        {"crew_id": "CRW-EL-02", "crew_name": "Viman Nagar Grid Technicians", "depot_node": "DEPOT-VIM", "depot_lat": 18.5679, "depot_lon": 73.9143, "skill": "Electrical Crew", "capacity": 8, "shift_hours": 8.0, "speed_kmh": 28.0},
        {"crew_id": "CRW-SN-01", "crew_name": "Central Pune Sanitation Unit", "depot_node": "DEPOT-SHI", "depot_lat": 18.5314, "depot_lon": 73.8446, "skill": "Sanitation Crew", "capacity": 10, "shift_hours": 8.0, "speed_kmh": 20.0},
        {"crew_id": "CRW-SN-02", "crew_name": "East Zone Waste Fleet", "depot_node": "DEPOT-HAD", "depot_lat": 18.5089, "depot_lon": 73.9259, "skill": "Sanitation Crew", "capacity": 9, "shift_hours": 8.0, "speed_kmh": 20.0},
    ]
    return pd.DataFrame(crews_data)


def generate_road_network() -> Dict[str, Any]:
    """Generate a connected road network for Pune municipal region."""
    nodes = {
        "DEPOT-KOT": {"lat": 18.5074, "lon": 73.8077, "label": "Kothrud Depot", "zone": "Kothrud"},
        "DEPOT-SHI": {"lat": 18.5314, "lon": 73.8446, "label": "Shivajinagar Depot", "zone": "Shivajinagar"},
        "DEPOT-HAD": {"lat": 18.5089, "lon": 73.9259, "label": "Hadapsar Depot", "zone": "Hadapsar"},
        "DEPOT-BAN": {"lat": 18.5590, "lon": 73.7868, "label": "Baner Depot", "zone": "Baner"},
        "DEPOT-SWA": {"lat": 18.5018, "lon": 73.8586, "label": "Swargate Depot", "zone": "Swargate"},
        "DEPOT-VIM": {"lat": 18.5679, "lon": 73.9143, "label": "Viman Nagar Depot", "zone": "Viman Nagar"},
        "INT-DECCAN": {"lat": 18.5170, "lon": 73.8410, "label": "Deccan Gymkhana", "zone": "Shivajinagar"},
        "INT-FC": {"lat": 18.5236, "lon": 73.8418, "label": "FC Road Chowk", "zone": "Shivajinagar"},
        "INT-JM": {"lat": 18.5280, "lon": 73.8510, "label": "JM Road Jn", "zone": "Shivajinagar"},
        "INT-UNIV": {"lat": 18.5528, "lon": 73.8260, "label": "Pune University Circle", "zone": "Baner"},
        "INT-AUNDH": {"lat": 18.5602, "lon": 73.8070, "label": "Aundh Chowk", "zone": "Baner"},
        "INT-MAGAR": {"lat": 18.5140, "lon": 73.9280, "label": "Magarpatta City Gate", "zone": "Hadapsar"},
        "INT-PULGATE": {"lat": 18.5080, "lon": 73.8820, "label": "Pulgate Camp", "zone": "Swargate"},
        "INT-KATRAJ": {"lat": 18.4575, "lon": 73.8677, "label": "Katraj Snake Park Jn", "zone": "Swargate"},
        "INT-KALYANI": {"lat": 18.5470, "lon": 73.9020, "label": "Kalyani Nagar Bridge", "zone": "Viman Nagar"},
        "INT-YERAWADA": {"lat": 18.5530, "lon": 73.8790, "label": "Yerawada Jn", "zone": "Viman Nagar"},
    }

    # Interconnected edges: (source, target, road_name, distance_km)
    edges = [
        ("DEPOT-KOT", "INT-DECCAN", "Karve Road", 4.2),
        ("DEPOT-BAN", "INT-AUNDH", "Baner-Aundh Road", 3.1),
        ("INT-AUNDH", "INT-UNIV", "Ganeshkhind Road", 2.8),
        ("INT-UNIV", "INT-FC", "Shivajinagar Link", 3.5),
        ("INT-FC", "INT-JM", "Modern College Rd", 1.2),
        ("INT-JM", "DEPOT-SHI", "JM Road North", 0.9),
        ("INT-DECCAN", "INT-FC", "FC Road South", 1.1),
        ("INT-DECCAN", "DEPOT-SWA", "Tilak Road", 2.7),
        ("DEPOT-SWA", "INT-PULGATE", "Shankarsheth Road", 3.0),
        ("DEPOT-SWA", "INT-KATRAJ", "Satara Road", 5.8),
        ("INT-PULGATE", "DEPOT-HAD", "Solapur Road", 5.2),
        ("DEPOT-HAD", "INT-MAGAR", "Magarpatta Road", 1.5),
        ("DEPOT-SHI", "INT-YERAWADA", "Sangamwadi Bridge", 3.8),
        ("INT-YERAWADA", "INT-KALYANI", "Nagar Road West", 2.9),
        ("INT-KALYANI", "DEPOT-VIM", "Kalyani-Viman Rd", 2.5),
        ("INT-UNIV", "INT-YERAWADA", "Airport Bypass", 6.2),
        ("INT-MAGAR", "INT-KALYANI", "Mundhwa Bridge", 4.6),
        ("DEPOT-KOT", "INT-UNIV", "Paud Road to Senapati Bapat", 6.4),
    ]

    edge_list = []
    for u, v, name, dist in edges:
        edge_list.append({
            "source": u,
            "target": v,
            "road_name": name,
            "distance_km": dist,
            "speed_kmh": 30.0,
            "travel_time_min": round((dist / 30.0) * 60, 2),
        })
        # Undirected road graph
        edge_list.append({
            "source": v,
            "target": u,
            "road_name": name,
            "distance_km": dist,
            "speed_kmh": 30.0,
            "travel_time_min": round((dist / 30.0) * 60, 2),
        })

    return {"nodes": nodes, "edges": edge_list}


def generate_full_dataset(
    target_count: int = 8000,
    seed: int = 42,
    raw_csv_path: str | Path = "data/raw/pune_civic_complaints.csv",
    crews_csv_path: str | Path = "data/raw/crews.csv",
    road_json_path: str | Path = "data/raw/road_network.json",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate and write the full reproducible synthetic dataset."""
    rng = np.random.default_rng(seed)
    py_rand = random.Random(seed)

    records = []
    for i in range(target_count):
        rec = generate_complaint_record(i, rng, py_rand, add_defects=True)
        records.append(rec)

    # Inject a controlled 0.5% duplicate set for preprocessing verification
    dup_count = int(target_count * 0.005)
    for k in range(dup_count):
        dup_record = records[k].copy()
        records.append(dup_record)

    df_complaints = pd.DataFrame(records)
    crews_df = generate_crews()
    road_network = generate_road_network()

    Path(raw_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(crews_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(road_json_path).parent.mkdir(parents=True, exist_ok=True)

    df_complaints.to_csv(raw_csv_path, index=False)
    crews_df.to_csv(crews_csv_path, index=False)
    with open(road_json_path, "w", encoding="utf-8") as f:
        json.dump(road_network, f, indent=2)

    return df_complaints, crews_df, road_network


if __name__ == "__main__":
    df, crews, road = generate_full_dataset()
    print(f"Dataset generated successfully! Complaints shape: {df.shape}, Crews: {crews.shape}, Road nodes: {len(road['nodes'])}")
