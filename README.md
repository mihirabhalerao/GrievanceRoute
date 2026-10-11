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

The application integrates classical artificial intelligence heuristic search, supervised learning, unsupervised learning, and combinatorial local search into a functional, 7-page Streamlit web application.

---

## 🎓 Syllabus & Lab Exercise Mapping

| Syllabus Unit | Topics Covered | GrievanceRoute Module | GUI Page |
| :--- | :--- | :--- | :--- |
| **Unit I: Introduction to AI** | Problem formulation, state space, production rules, PEAS agent | `src/grievanceroute/services/` | `02 Complaint Management` |
| **Unit II: Heuristic Search** | A* Search ($f=g+h$ with admissible Euclidean lower bound), Hill Climbing (2-opt), Simulated Annealing | `src/grievanceroute/ai/` | `05 Crew Route Planner` |
| **Unit III: ML & Preprocessing** | Deduplication, mode imputation, coordinate validation, casing normalization, TF-IDF, scaling | `src/grievanceroute/data/` | `07 Data Quality & Insights` |
| **Unit IV: Supervised Learning** | Logistic Regression (Primary), Decision Tree, Multinomial Naive Bayes, Majority Baseline, Linear/RF Regressors vs Median Baseline | `src/grievanceroute/ml/` | `04 ML Workbench`<br/>`01 Operations Dashboard` |
| **Unit V: Unsupervised Learning** | K-Means (projected metric coordinates), DBSCAN density clustering, Apriori association rules | `src/grievanceroute/ml/` | `03 Geographic Hotspots`<br/>`06 Experiments` |
| **Unit VI: Optimization** | Constrained multi-crew VRP objective $J(R)$, 2-opt search, Simulated Annealing cooling schedule | `src/grievanceroute/routing/`<br/>`src/grievanceroute/ai/` | `05 Crew Route Planner`<br/>`06 Experiments` |

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
│   ├── 02_Complaint_Management.py      # Validated citizen intake, triage, & audit history
│   ├── 03_Geographic_Hotspots.py       # K-Means & DBSCAN density maps on projected coordinates
│   ├── 04_ML_Workbench.py              # Classifier & regressor benchmarks, ROC curves, Error Analysis
│   ├── 05_Crew_Route_Planner.py        # Multi-crew VRP with Greedy, 2-Opt HC, and Simulated Annealing
│   ├── 06_Experiments.py               # MLflow tracking, metrics CSVs, artifact registry
│   └── 07_Data_Quality.py              # Preprocessing demos, scaling, sampling, defect audits
├── src/grievanceroute/
│   ├── data/                           # Generator, validator, preprocessor, feature engineering
│   ├── ml/                             # Classifiers, regressors, clustering, Apriori association
│   ├── ai/                             # A* search (Euclidean heuristic), 2-opt HC, Simulated Annealing
│   ├── routing/                        # NetworkX graph, shift constraints, VRP solvers
│   ├── database/                       # SQLite schema DDL, repository CRUD
│   └── services/                       # Triage prediction, complaint intake, route planning
├── artifacts/
│   ├── models/                         # Serialized Joblib .pkl pipelines
│   ├── metrics/                        # Benchmark JSONs and exported CSVs
│   └── figures/                        # Generated architecture, workflow, and ROC plots
├── tests/                              # Pytest test suite (21 unit, integration, & workflow tests)
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
You can generate data, train models, run experiments, and initialize the database in sequence:
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
All 21 tests pass with 0 failures, covering dataset generation reproducibility, defect cleaning, target leakage prevention, coordinate projection, classifier benchmarks against Majority Baseline, regressor benchmarks against Median Baseline, spatial clustering, Apriori rule mining, A* admissibility, 2-opt hill climbing, simulated annealing, shift budget constraints, safety overrides, and end-to-end integration workflows.

---

## 📊 Measured Benchmark Results

All figures below are exact empirical results measured across our 8,000-record dataset (80/20 train/test split, 1,600 test records):

### 1. Classification (Test Split: 1,600 Records)
| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | 26.87% | 0.1059 | 0.1139 | 0.0672 | 0.2500 | 0.20s |
| **Logistic Regression (Primary)** | **96.37%** | **0.9635** | **0.9638** | **0.9633** | **0.9638** | **0.34s** |
| Multinomial Naive Bayes | 96.37% | 0.9635 | 0.9638 | 0.9633 | 0.9638 | 0.22s |
| Decision Tree (depth=15) | 96.25% | 0.9622 | 0.9625 | 0.9620 | 0.9625 | 0.44s |

*Misclassification & Error Analysis*: Realistic cross-domain citizen filings (e.g. road damage caused by water pipeline bursts or trash piles around light poles) produce ~3.6% natural boundary misclassifications, visible and diagnosable in the **Error Analysis** workbench tab.

### 2. Multi-Target Regression (Test Split: 1,600 Records)
| Target | Model | MAE | RMSE | $R^2$ Score | Mean Residual | Std Residual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Urgency Score (1–5)** | **Median Baseline** | 0.8475 | 1.0583 | -0.0430 | +0.2006 | 1.0391 |
| | Linear Regression | 0.7912 | 0.9372 | 0.1820 | +0.0076 | 0.9371 |
| | **Random Forest Regressor** | **0.7656** | **0.9199** | **0.2120** | **+0.0093** | **0.9199** |
| **Resolution Time (Hours)** | **Median Baseline** | 12.0958 hrs | 14.8790 hrs | -0.0032 | -0.2131 | 14.8775 |
| | Linear Regression | 11.3032 hrs | 13.7888 hrs | 0.1411 | -0.2608 | 13.7863 |
| | **Random Forest Regressor** | **11.0085 hrs** | **13.5797 hrs** | **0.1670** | **-0.2114** | **13.5781** |

### 3. Spatial Hotspot Clustering
- **K-Means ($K=5$)**: Silhouette Score = **0.4864**, Inertia = **46,768.5** computed on metric projected planar coordinates $(x_{\text{km}}, y_{\text{km}})$.
- **DBSCAN ($\epsilon=0.8\text{ km}$, $\text{min\_samples}=25$)**: Detects dense grievance corridors with 2.9% noise fraction.

### 4. Vehicle Routing Problem (VRP) Optimization (8 Specialized Municipal Crews)
| Algorithm | Total Distance | Distance Imprv (%) | Objective Cost $J(R)$ | Complaints Served | Total Overtime | Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Greedy Baseline** | 270.95 km | 0.00% | 370.95 | 41 / 45 | 0.00 hrs | 104.4 ms |
| **2-Opt Hill Climbing** | **194.85 km** | **+28.09%** | **274.85** | 41 / 45 | 0.00 hrs | 7.1 ms |
| **Simulated Annealing** | **196.54 km** | **+27.46%** | **276.54** | 41 / 45 | 0.00 hrs | 10.1 ms |

*Route Feasibility Summary*: Both local search heuristics achieve substantial travel savings (~28% distance reduction) while strictly satisfying the hard 8.0-hour shift limit (0.0 hours overtime) and crew specialization constraints.

---

## ⚠️ Limitations & Methodological Disclosure
1. **Synthetic Nature of Dataset**: The 8,000 complaints are generated using probabilistic sentence templates with simulated defects (duplicates, missing weather, anomalous coordinates). The near-perfect classification metrics on synthetic test splits reflect template consistency rather than organic citizen variability.
2. **Deterministic Rules & Lexicons**: The electrical safety override relies on substring phrase matching; semantic expressions not included in the keyword lexicon require supervisor vigilance.
3. **Road Graph Approximation**: The current network uses 16 major Pune junction nodes with Euclidean $A^*$ lower bounds rather than a complete OpenStreetMap road network.
