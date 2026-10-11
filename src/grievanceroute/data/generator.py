"""
Realistic Synthetic Dataset Generator for GrievanceRoute.
Generates reproducible Pune civic complaints with realistic linguistic variation,
informal Hinglish phrasing, cross-domain ambiguity, and configurable data-quality defects.
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

# Rich varied templates with colloquial, abbreviation, and informal phrasing
TEXT_TEMPLATES = {
    "pothole": [
        "Deep dangerous pothole on the main road near {loc}, causing bike accidents.",
        "Sadak par bahut bada gaddha hai near {loc}, vehicles getting damaged daily.",
        "Road surface broken with multiple crater-like pot holes near {loc} chowk.",
        "Open trench and deep khadda on street opposite {loc} market.",
        "Severe asphalt damage and uneven road causing water stagnation near {loc}.",
        "Gaddhe pad gaye hain road pe at {loc}, two-wheelers slipping regularly.",
        "Large ditch in the middle of the bus lane near {loc} signal.",
        "Big crater formed after rains near {loc}. Immediate patch work required.",
        "Rasta kharab zala ahe, massive crater near {loc} causing heavy traffic jam.",
        "Broken tar surface and loose gravel creating hazard near {loc} corner.",
        "Road contractor left deep excavation unpaved at {loc} lane.",
        "Pothole repaired last week has sunk again near {loc} bridge.",
    ],
    "water leak": [
        "Main pipeline burst and clean drinking water is gushing onto the road at {loc}.",
        "Paani ki pipe line leak ho gayi hai near {loc}, thousands of liters wasted.",
        "Severe water logging due to underground drinking water pipe rupture near {loc} garden.",
        "Low water pressure and dirty tap supply reported across {loc} colony.",
        "Supply pipe cracked near storm drain at {loc}, entire street flooded with potable water.",
        "Nal se ganda paani aa raha hai and line leaking continuously at {loc}.",
        "Municipal valve chamber leaking water onto baseline near {loc} society.",
        "Continuous water leakage from main valve box at {loc} corner for 2 days.",
        "Drinking water line broken during cable laying work near {loc}.",
        "Water fountain sprouting from road due to high pressure pipe burst at {loc}.",
        "Pipeline leak at {loc} chowk reducing drinking water supply to all societies.",
        "Paani waha se beh raha hai continuously near {loc} school gate.",
    ],
    "streetlight": [
        "Dark street! All streetlights are completely off and dark near {loc} lane.",
        "Live electrical wire hanging dangerously from a bent light pole near {loc}.",
        "Streetlight pole sparking continuously near {loc} bus stand, danger of shock.",
        "Bijli ka pole band hai aur raat ko andhera rehta hai near {loc}.",
        "Exposed underground wires near the streetlight junction box at {loc}.",
        "Streetlight flickering violently and short-circuit sparks visible at {loc}.",
        "Complete blackout on municipal road stretch near {loc} school.",
        "Street lamps fused for past 5 days at {loc} bypass, unsafe for pedestrians.",
        "Pole light not working since Monday near {loc} chowk, please repair lamp.",
        "Streetlight timer broken, light remains off all night at {loc} colony.",
        "Dangerous exposed wiring on streetlight pole base outside {loc} bank.",
        "Rasteche dive band ahet near {loc}, area pitch dark after 7 pm.",
    ],
    "garbage": [
        "Huge pile of uncollected garbage rotting and stinking on the corner of {loc}.",
        "Kachra petti overflow ho rahi hai and stray cattle spreading it near {loc}.",
        "Solid waste dumped on footpath blocking pedestrians near {loc} market.",
        "Sanitation vehicle has not visited for four days; waste piling up at {loc}.",
        "Open dumping and burning of plastic garbage creating hazardous smoke at {loc}.",
        "Debris and waste unattended near public bin at {loc} colony.",
        "Overflowing dustbin with foul smell attracting flies near {loc} hospital.",
        "Garbage dumping spot created illegally near the corner at {loc}.",
        "Kachra gaadi nahi aayi since 3 days at {loc}, bins completely full.",
        "Dumping yard smell unbearable near residential gate at {loc}.",
        "Rotting organic waste spilled over road near {loc} vegetables market.",
        "Safai karmchari missing, waste heaps lying open along {loc} road.",
    ],
}

# Ambiguous and cross-cutting realistic citizen complaints
AMBIGUOUS_TEMPLATES = [
    ("pothole", "Water leaking from road side has filled deep potholes near {loc}, making ditch invisible."),
    ("water leak", "Pothole digging damaged underground drinking water line near {loc}, water gushing out."),
    ("garbage", "People throwing domestic garbage near the non-functional streetlight pole at {loc}."),
    ("streetlight", "Garbage collection vehicle hit the streetlight pole at {loc}, wires now exposed and dark."),
    ("pothole", "Huge crater in road near garbage container at {loc} causing traffic congestion."),
    ("water leak", "Road damaged where water pipeline valve was recently excavated near {loc}."),
]

# Realistic cross-domain misclassification cases (citizen misfiling / multi-hazard narrative)
# In these templates, the narrative is dominated by a secondary hazard, challenging pure n-gram bag-of-words
CROSS_DOMAIN_MISCLASSIFICATION_TEMPLATES = {
    "pothole": [
        "Underground municipal water pipeline burst and continuous leakage is flooding the road near {loc}.",
        "Water supply pipe broke near {loc} chowk and heavy water is flowing onto the entire street.",
        "Severe water logging due to open drainage line overflow near {loc}, road completely submerged.",
    ],
    "water leak": [
        "Deep dangerous road crater and uneven surface near {loc} causing two-wheelers to slip.",
        "Huge asphalt trench left open by road repair contractors near {loc} market.",
        "Road completely broken and loose tarmac stones scattered near {loc} junction.",
    ],
    "streetlight": [
        "Solid waste and plastic garbage piled up on the corner of {loc}, foul smell spreading.",
        "Overflowing kachra petti and stray cattle scattering garbage near {loc} lane.",
        "Illegal dumping of construction debris and rotting trash blocking the pathway at {loc}.",
    ],
    "garbage": [
        "All streetlights are completely off and dark stretch near {loc} after 8 PM, very unsafe.",
        "Municipal streetlight pole fused for a week near {loc} chowk, area pitch dark.",
        "Street lamps not working on the main lane near {loc} school, pedestrians in the dark.",
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
    add_defects: bool = True
) -> Dict[str, Any]:
    """Generate a single realistic complaint record with natural variance and noise."""
    # Natural slightly imperfect class distribution (pothole 27%, garbage 26%, water leak 24%, streetlight 23%)
    categories = ["pothole", "garbage", "water leak", "streetlight"]
    cat_weights = [0.27, 0.26, 0.24, 0.23]
    category = py_rand.choices(categories, weights=cat_weights)[0]

    zone = py_rand.choice(list(PUNE_ZONES.keys()))
    zone_info = PUNE_ZONES[zone]

    # Sample coordinates around zone center
    lat = float(rng.normal(loc=zone_info["lat"], scale=0.014))
    lon = float(rng.normal(loc=zone_info["lon"], scale=0.014))

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
    hist_base = {"pothole": 36.0, "water leak": 20.0, "streetlight": 24.0, "garbage": 14.0}[category]
    historical_resolution_hours = round(float(rng.normal(loc=hist_base, scale=5.0)), 1)
    historical_resolution_hours = max(4.0, historical_resolution_hours)

    # Text description selection
    loc_name = py_rand.choice(LOCALITY_NAMES)
    rand_val = py_rand.random()
    # 4% chance of realistic citizen category misfiling / compound hazard
    # (e.g., citizen filed under 'pothole' due to driving distress, but entered text describing a water pipe leak)
    cross_donor = {
        "pothole": "water leak",
        "water leak": "pothole",
        "streetlight": "garbage",
        "garbage": "streetlight",
    }
    if rand_val < 0.04:
        template = py_rand.choice(TEXT_TEMPLATES[cross_donor[category]])
    # 8% chance of picking a cross-domain ambiguous template
    elif rand_val < 0.12:
        matching_amb = [t for t in AMBIGUOUS_TEMPLATES if t[0] == category]
        template = py_rand.choice(matching_amb)[1] if matching_amb else py_rand.choice(TEXT_TEMPLATES[category])
    else:
        template = py_rand.choice(TEXT_TEMPLATES[category])

    complaint_text = template.format(loc=loc_name)

    # Introduce natural colloquial variations / typos (5% chance)
    if py_rand.random() < 0.05:
        typo_map = {"pothole": "pot hole", "pipeline": "pipe line", "light": "lite", "garbage": "garbge"}
        for k, v in typo_map.items():
            complaint_text = complaint_text.replace(k, v)

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
        if "hospital" in complaint_text.lower():
            urgency_raw += 1.6
        if repeat:
            urgency_raw += 0.6

    if repeat:
        urgency_raw += 0.7
    if weather == "Stormy":
        urgency_raw += 0.6

    # Real-world noise on urgency
    urgency_raw += float(rng.normal(0, 0.45))
    urgency_score = int(np.clip(round(urgency_raw), 1, 5))

    # Actual resolution hours (skewed distribution)
    base_time = {1: 52.0, 2: 38.0, 3: 26.0, 4: 16.0, 5: 8.0}[urgency_score]
    delay = 1.0
    if weather in ["Rainy", "Stormy"]:
        delay += 0.35
    if repeat:
        delay += 0.2

    # Lognormal noise with occasional backlog spike
    actual_hours = base_time * delay + float(rng.lognormal(mean=1.8, sigma=0.6))
    actual_resolution_hours = round(float(np.clip(actual_hours, 1.5, 168.0)), 1)

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

    # Configurable realistic defects for preprocessing demonstration
    if add_defects:
        prob = py_rand.random()
        if prob < 0.02:
            # Missing weather
            record["weather"] = None
        elif prob < 0.035:
            # Extra leading / trailing whitespace
            record["complaint_text"] = "  " + record["complaint_text"] + "   "
        elif prob < 0.05:
            # Inconsistent casing in zone
            record["zone"] = record["zone"].lower()
        elif prob < 0.055:
            # Slight coordinate anomaly (outside Pune primary bounding box)
            record["latitude"] = round(lat + 0.3, 6)

    return record


def generate_crews() -> pd.DataFrame:
    """Generate crews configuration dataframe."""
    crews_data = [
        {"crew_id": "CRW-RD-01", "crew_name": "Kothrud Road Works", "depot_node": "DEPOT-KOT", "depot_lat": 18.5074, "depot_lon": 73.8077, "skill": "Road Maintenance", "capacity": 7, "shift_hours": 8.0, "speed_kmh": 25.0},
        {"crew_id": "CRW-RD-02", "crew_name": "Hadapsar Asphalt Team", "depot_node": "DEPOT-HAD", "depot_lat": 18.5089, "depot_lon": 73.9259, "skill": "Road Maintenance", "capacity": 6, "shift_hours": 8.0, "speed_kmh": 25.0},
        {"crew_id": "CRW-WS-01", "crew_name": "Shivajinagar Hydro Squad", "depot_node": "DEPOT-SHI", "depot_lat": 18.5314, "depot_lon": 73.8446, "skill": "Water Supply Crew", "capacity": 6, "shift_hours": 8.0, "speed_kmh": 22.0},
        {"crew_id": "CRW-WS-02", "crew_name": "Swargate Pipeline Response", "depot_node": "DEPOT-SWA", "depot_lat": 18.5018, "depot_lon": 73.8586, "skill": "Water Supply Crew", "capacity": 5, "shift_hours": 8.0, "speed_kmh": 22.0},
        {"crew_id": "CRW-EL-01", "crew_name": "Baner Power & Lighting", "depot_node": "DEPOT-BAN", "depot_lat": 18.5590, "depot_lon": 73.7868, "skill": "Electrical Crew", "capacity": 8, "shift_hours": 8.0, "speed_kmh": 28.0},
        {"crew_id": "CRW-EL-02", "crew_name": "Viman Nagar Grid Technicians", "depot_node": "DEPOT-VIM", "depot_lat": 18.5679, "depot_lon": 73.9143, "skill": "Electrical Crew", "capacity": 8, "shift_hours": 8.0, "speed_kmh": 28.0},
        {"crew_id": "CRW-SN-01", "crew_name": "Central Pune Sanitation Unit", "depot_node": "DEPOT-SHI", "depot_lat": 18.5314, "depot_lon": 73.8446, "skill": "Sanitation Crew", "capacity": 9, "shift_hours": 8.0, "speed_kmh": 20.0},
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
        "INT-KATRAJ": {"lat": 18.4575, "lon": 73.8677, "label": "Katraj Chowk", "zone": "Swargate"},
        "INT-KALYANI": {"lat": 18.5470, "lon": 73.9020, "label": "Kalyani Nagar Bridge", "zone": "Viman Nagar"},
        "INT-YERAWADA": {"lat": 18.5530, "lon": 73.8790, "label": "Yerawada Jn", "zone": "Viman Nagar"},
    }

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
            "source": u, "target": v, "road_name": name,
            "distance_km": dist, "speed_kmh": 30.0,
            "travel_time_min": round((dist / 30.0) * 60, 2),
        })
        edge_list.append({
            "source": v, "target": u, "road_name": name,
            "distance_km": dist, "speed_kmh": 30.0,
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
    """Generate and write the full reproducible synthetic dataset with realistic defects."""
    rng = np.random.default_rng(seed)
    py_rand = random.Random(seed)

    records = []
    for i in range(target_count):
        rec = generate_complaint_record(i, rng, py_rand, add_defects=True)
        records.append(rec)

    # Controlled exact duplicates (0.5%)
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
