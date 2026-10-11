# GrievanceRoute: Intelligent Civic Complaint Triage and Crew Routing System

[![Tests](https://github.com/mihirabhalerao/GrievanceRoute/actions/workflows/tests.yml/badge.svg)](https://github.com/mihirabhalerao/GrievanceRoute/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AIML Laboratory Mini-Project**  
> Group 37 | Batch B4 — Third Year B.Tech Computer Engineering  
> MKSSS's Cummins College of Engineering for Women, Pune  
> Affiliated to Savitribai Phule Pune University (SPPU)

---

## 📌 Overview

**GrievanceRoute** is an end-to-end, reproducible, intelligent civic complaint triage and repair crew routing system. Built for municipal operations in Pune, Maharashtra, it addresses the challenges of high-volume citizen complaints through automated Natural Language Processing (NLP), multi-target regression, spatial hotspot clustering, and constrained vehicle routing optimization (VRP).

The application integrates classical artificial intelligence heuristic search, supervised learning, unsupervised learning, and combinatorial optimization into a functional, multi-page Streamlit web application.

---

## 🎓 Syllabus & Lab Exercise Mapping

| Syllabus Unit | Topics Covered | GrievanceRoute Module | GUI Page |
| :--- | :--- | :--- | :--- |
| **Unit I: Introduction to AI** | Problem formulation, state space, production systems, PEAS agent | `src/grievanceroute/services/` | `02 Complaint Management` |
| **Unit II: Heuristic Search** | A*, Greedy Best-First, Hill Climbing, Simulated Annealing, Water-Jug, 8-Puzzle, AO*, Minimax | `src/grievanceroute/ai/` | `07 AI Algorithm Lab`<br/>`05 Crew Route Planner` |
| **Unit III: ML & Preprocessing** | Deduplication, imputation, TF-IDF, scaling, binning, sampling | `src/grievanceroute/data/` | `08 Data Quality & Insights` |
| **Unit IV: Supervised Learning** | Logistic Regression, Decision Tree, KNN, Naive Bayes, Linear & Random Forest Regressors | `src/grievanceroute/ml/` | `04 ML Workbench`<br/>`01 Operations Dashboard` |
| **Unit V: Unsupervised Learning** | K-Means (projected coordinates), DBSCAN, Hierarchical clustering, Apriori association rules | `src/grievanceroute/ml/` | `03 Geographic Hotspots`<br/>`06 Experiments` |
| **Unit VI: Optimization** | Multi-crew VRP objective $J(R)$, 2-opt search, SGD/Adam/RMSprop comparison | `src/grievanceroute/routing/`<br/>`src/grievanceroute/ml/neural_net.py` | `05 Crew Route Planner`<br/>`06 Experiments` |

---

## 🏗️ Repository Layout

```text
GrievanceRoute/
├── README.md                           # Comprehensive documentation & guide
├── requirements.txt                    # Pinned package dependencies
├── pyproject.toml                      # Build metadata & project configuration
├── Dockerfile                          # Containerized deployment definition
├── Makefile                            # Developer automation commands
├── app.py                              # Main Streamlit portal entry point
├── config/
│   ├── settings.yaml                   # Global application configuration
│   └── feature_schema.yaml             # Schema definitions and data types
├── data/
│   ├── raw/                            # pune_civic_complaints.csv, crews.csv, road_network.json
│   ├── processed/                      # cleaned_complaints.csv
│   └── grievance_route.db              # SQLite operational database
├── pages/
│   ├── 01_Operations_Dashboard.py      # Real-time KPIs, queues, category/zone analytics
│   ├── 02_Complaint_Management.py      # Validated citizen intake & audit history
│   ├── 03_Geographic_Hotspots.py       # K-Means & DBSCAN density maps on projected coordinates
│   ├── 04_ML_Workbench.py              # Classifier & regressor benchmarks, ROC curves
│   ├── 05_Crew_Route_Planner.py        # Multi-crew VRP with Greedy, HC, and Simulated Annealing
│   ├── 06_Experiments.py               # MLflow tracking, metrics CSVs, artifact registry
│   ├── 07_AI_Algorithm_Lab.py          # A*, Best-First, Water-Jug, 8-Puzzle, AO*, Minimax
│   └── 08_Data_Quality.py              # Preprocessing demos, scaling, sampling, binning
├── src/grievanceroute/
│   ├── data/                           # Generator, validator, preprocessor, feature engineering
│   ├── ml/                             # Classifiers, regressors, clustering, Apriori, neural net
│   ├── ai/                             # A*, best-first, hill climbing, annealing, classic demos
│   ├── routing/                        # NetworkX graph, shift constraints, VRP solvers
│   ├── database/                       # SQLite schema DDL, repository CRUD
│   └── services/                       # Triage prediction, complaint intake, route planning
├── artifacts/
│   ├── models/                         # Serialized Joblib .pkl pipelines
│   ├── metrics/                        # Benchmark JSONs and exported CSVs
│   └── figures/                        # Generated architecture, workflow, and ROC plots
├── tests/                              # Pytest test suite (18 unit & integration tests)
└── docs/                               # Architecture, algorithms, dataset card, report outline
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/mihirabhalerao/GrievanceRoute.git
cd GrievanceRoute
python -m venv venv

# Windows activate:
.\venv\Scripts\activate
# Linux/macOS activate:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Pipeline Automation
You can generate data, train models, run experiments, and initialize the database in one sequence:
```bash
# Using Makefile
make setup

# Or running the standalone Python scripts:
python scripts/generate_dataset.py
python scripts/validate_dataset.py
python scripts/preprocess_data.py
python scripts/train_models.py
python scripts/run_experiments.py
python scripts/initialize_database.py
python scripts/generate_report_assets.py
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full pytest suite with verbose output:
```bash
pytest tests/ -v
```
All 18 tests cover dataset generation, leakage prevention, classifier pipelines, clustering, Apriori, A* optimality, water-jug, 8-puzzle solvability, AO*, minimax, VRP constraints, and SQLite persistence.

---

## 📊 Measured Benchmark Results

### 1. Classification (Untouched Test Split: 1,600 Records)
| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Primary)** | **100.0%** | **1.0000** | **1.0000** | **1.0000** | **0.26s** |
| Decision Tree (depth=15) | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0.22s |
| K-Nearest Neighbors (k=5) | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0.22s |
| Multinomial Naive Bayes | 100.0% | 1.0000 | 1.0000 | 1.0000 | 0.21s |

### 2. Multi-Target Regression
| Target | Model | MAE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Urgency Score (1–5)** | Random Forest | **0.7908** | **0.9181** | **0.2206** |
| | Linear Regression | 0.8259 | 0.9396 | 0.1838 |
| **Actual Resolution Hours** | Random Forest | **11.6152 hrs** | **14.0156 hrs** | **0.1450** |
| | Linear Regression | 11.9858 hrs | 14.2366 hrs | 0.1178 |

### 3. Spatial Clustering Hotspots
- **K-Means ($K=5$)**: Silhouette Score = **0.5450**, Inertia = **33,804.1** (Projected planar coordinates in km).
- **DBSCAN ($\epsilon=0.8\text{ km}$, $\text{min\_samples}=25$)**: Discovers 2 high-density zones with 1.1% noise points.

### 4. Vehicle Routing Problem (VRP) Optimization
| Algorithm | Distance (km) | Distance Imprv (%) | Objective Cost $J(R)$ | Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy Baseline** | 98.4 km | 0.0% | 185.4 | 66.1 ms |
| **2-Opt Hill Climbing** | 84.1 km | +14.5% | 158.2 | 4.5 ms |
| **Simulated Annealing** | **80.2 km** | **+18.5%** | **150.1** | **11.9 ms** |

---

## ⚠️ Limitations & Disclosure
1. **Synthetic Nature of Dataset**: The 8,000 complaints are generated using probabilistic sentence templates. The perfect classification metrics on the random split reflect template consistency rather than real-world generalizability.
2. **Deterministic Rules**: The electrical hazard override relies on substring phrase matching; semantic variations not included in the lexicon require operator review.
3. **Road Graph Approximation**: The network uses 16 major Pune junction nodes rather than a complete OpenStreetMap road graph.
