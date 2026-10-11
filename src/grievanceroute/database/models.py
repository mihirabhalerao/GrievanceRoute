"""
Database Schema Definitions for GrievanceRoute.
Defines SQL table DDL for complaints, status history, crews, and route plans.
"""

CREATE_COMPLAINTS_TABLE = """
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    complaint_text TEXT NOT NULL,
    complaint_type TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    zone TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    hour INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    weather TEXT,
    monsoon INTEGER NOT NULL DEFAULT 0,
    low_lying_area INTEGER NOT NULL DEFAULT 0,
    repeat_complaint INTEGER NOT NULL DEFAULT 0,
    historical_resolution_hours REAL,
    urgency_score INTEGER NOT NULL,
    actual_resolution_hours REAL,
    status TEXT NOT NULL DEFAULT 'Pending',
    crew_required TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_STATUS_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id TEXT NOT NULL,
    old_status TEXT NOT NULL,
    new_status TEXT NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY(complaint_id) REFERENCES complaints(complaint_id) ON DELETE CASCADE
);
"""

CREATE_CREWS_TABLE = """
CREATE TABLE IF NOT EXISTS crews (
    crew_id TEXT PRIMARY KEY,
    crew_name TEXT NOT NULL,
    depot_node TEXT NOT NULL,
    depot_lat REAL NOT NULL,
    depot_lon REAL NOT NULL,
    skill TEXT NOT NULL,
    capacity INTEGER NOT NULL,
    shift_hours REAL NOT NULL,
    speed_kmh REAL NOT NULL
);
"""

CREATE_ROUTES_TABLE = """
CREATE TABLE IF NOT EXISTS route_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_date TEXT NOT NULL,
    crew_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    distance_km REAL NOT NULL,
    jobs_count INTEGER NOT NULL,
    stops_json TEXT NOT NULL,
    cost REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
