# Dataset Card: Pune Civic Complaints Synthetic Dataset

## 1. Dataset Summary
The primary dataset for GrievanceRoute contains **8,040 raw synthetic civic complaints** (cleaned to **8,000 records**) modeled after urban grievances encountered in Pune, Maharashtra. It includes companion configurations for 8 specialized municipal repair crews (`crews.csv`) and a 16-node weighted road graph topology (`road_network.json`).

All data generation is strictly reproducible using a fixed random seed (`seed = 42`).

---

## 2. Feature Schema & Field Descriptions

| Column | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `complaint_id` | String | Identifier | Unique tracking code (e.g., `PUN-GRV-00001`) |
| `complaint_text` | String | NLP Input | Free-text citizen narrative in formal English, colloquial Hinglish, and informal styles |
| `complaint_type` | Categorical | Target (Classification) | `pothole`, `water leak`, `streetlight`, `garbage` |
| `latitude` | Float | Spatial Input | Simulated latitude (18.40 – 18.65) |
| `longitude` | Float | Spatial Input | Simulated longitude (73.70 – 74.00) |
| `zone` | Categorical | Spatial Input | `Kothrud`, `Shivajinagar`, `Hadapsar`, `Baner`, `Swargate`, `Viman Nagar` |
| `timestamp` | Datetime | Temporal Input | Arrival time of citizen grievance |
| `hour` | Integer | Temporal Input | Hour of arrival (0–23) |
| `day_of_week` | Integer | Temporal Input | Day index (0=Monday, 6=Sunday) |
| `weather` | Categorical | Context Input | `Clear`, `Rainy`, `Cloudy`, `Stormy` |
| `monsoon` | Boolean | Context Input | Active monsoon flag |
| `low_lying_area` | Boolean | Context Input | Topographical flood risk indicator |
| `repeat_complaint` | Boolean | Context Input | Prior unresolved occurrence indicator |
| `historical_resolution_hours` | Float | Historical Context | Average historical repair time for zone & issue |
| `urgency_score` | Integer (1–5) | Target (Regression) | Severity index (1=Minor, 5=Emergency) |
| `actual_resolution_hours` | Float | Target (Regression) | Realized resolution duration (1.5 – 168.0 hrs) |
| `status` | Categorical | Metadata | `Pending`, `Assigned`, `In Progress`, `Resolved` |
| `crew_required` | Categorical | Operational Metadata | Specialized skill needed |

---

## 3. Linguistic Diversity & Preprocessing Defects
To provide a realistic municipal NLP challenge and support Unit III laboratory exercises, the generator introduces:
- **Linguistic Phrasing**:
  - Formal English: *"Massive water pipe burst near chowk causing severe flooding."*
  - Colloquial Hinglish: *"Bahut bada water pipeline foot gaya hai paani waste ho raha hai."*
  - Informal Phrasing: *"Huge crater on main road, 2 bikes skidded today, very risky!"*
  - Cross-Domain Nuance: *"Water logging due to open pothole"* (multi-keyword challenge).
- **Controlled Injected Defects**:
  - **Duplicates**: 40 records (0.5%) injected with duplicate complaint IDs and descriptions.
  - **Missing Values**: 184 null records in the `weather` feature for mode imputation.
  - **Coordinate Outliers**: 8 records with anomalous coordinates outside Pune boundaries.
  - **Categorical Inconsistency**: Inconsistent zone casing (e.g., `kothrud`, `HADAPSAR `).
  - **Skewed Backlog**: 20 long-tail backlog cases exceeding 100 hours.

The cleaning pipeline (`src/grievanceroute/data/preprocessor.py`) removes duplicates, normalizes casing, drops coordinate anomalies, and imputes missing weather values cleanly.

---

## 4. Leakage Prevention Protocol
The following features are **strictly prohibited** from serving as inputs during model training:
- `complaint_type`: Ground truth classification target.
- `urgency_score`: Ground truth urgency target.
- `actual_resolution_hours`: Future post-resolution variable.
- `status`: Dynamic operational state unknown at intake time.

All training features are verified to contain exclusively information available at initial intake.

---

## 5. Limitations & Ethical Considerations
- **Synthetic Nature**: The dataset is generated from diverse sentence templates and normal spatial distributions. It does not reflect official municipal records from the Pune Municipal Corporation (PMC).
- **Evaluation Caveat**: High test accuracy across template variants does not guarantee generalizability to real-world, noisy municipal citizen feedback. The system includes an Error Analysis workbench tab to audit model failures.
- **Privacy**: No real citizen names, contact numbers, or private residential addresses are utilized.
