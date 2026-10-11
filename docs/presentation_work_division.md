# GrievanceRoute — Presentation Work Division & Role Mapping
## Third Year B. Tech (Computer Engineering) — Group 37 (Batch B4)
### MKSSS’s Cummins College of Engineering for Women, Pune | Academic Year 2026–27

---

## 📌 Executive Presentation Strategy

The project presentation is designed as an interactive, coordinated 20-minute defense (15 minutes presentation + 5 minutes live demonstration & viva voce) divided among **4 team members**. 

Every team member has an equal share of core computer engineering and AIML responsibilities:
- **Person 1**: Problem Formulation, Synthetic Data Engineering, Data Hygiene & Preprocessing Lab.
- **Person 2**: NLP Text Classification, Supervised Regression, Baselines & Error Analysis.
- **Person 3**: Spatial Intelligence (K-Means & DBSCAN), Association Rule Mining & Operations Dashboard.
- **Person 4**: Combinatorial Heuristic Search ($A^*$, 2-Opt, Simulated Annealing), VRP Routing & Full-Stack MLOps.

---

## 👥 Person-by-Person Mapping Summary

| Person | Primary Role / Title | Focus Modules | Key Algorithms / Mathematical Concepts | Assigned GUI Pages |
| :---: | :--- | :--- | :--- | :--- |
| **Person 1** | **Data Architect & Pipeline Engineer** | Data Generation, Ingestion, Preprocessing & Quality | Equirectangular Planar Projection, Mode Imputation, Statistical Sampling (SRSWOR, SRSWR, Stratified, Cluster), Discretization | `07 Data Quality & Insights` |
| **Person 2** | **NLP & Supervised Learning Engineer** | Text Classification, Urgency & Duration Regression | Sublinear TF-IDF, Multinomial Logistic Regression, Decision Trees, Random Forest Regressors, Majority & Median Baselines | `04 ML Workbench` |
| **Person 3** | **Spatial Intelligence & Analytics Engineer** | Geospatial Hotspots, Pattern Mining & Operations | Projected K-Means (Silhouette & Inertia), DBSCAN Density Clustering, Custom Apriori Rule Mining (Support, Confidence, Lift) | `01 Operations Dashboard`<br/>`03 Geographic Hotspots` |
| **Person 4** | **Algorithms, Routing & Systems Engineer** | VRP Routing, Heuristic Search, System Lifecycle & CI | Admissible $A^*$ Search, 2-Opt Hill Climbing, Simulated Annealing (Cooling Schedule), Safety Overrides, SQLite CRUD | `App Portal`<br/>`02 Complaint Management`<br/>`05 Crew Route Planner`<br/>`06 Experiments` |

---

## 📋 Detailed Individual Assignment & File Ownership

### 👤 Person 1: Data Architect & Pipeline Engineer
*“Designing the Municipal Data Foundation, Realistic Linguistic Variations, and Quality Engineering”*

#### 1. Core Responsibilities
- Explain the real-world civic governance problem in Pune, municipal ward demographics, and the research gap identified in literature survey.
- Explain the data generation pipeline, the synthetic data design rationale, and why probabilistic templates with Hinglish/colloquial phrasing were used.
- Explain the data quality defects injected for Unit III syllabus coverage (duplicate tickets, missing weather values, coordinate anomalies, and casing inconsistencies).
- Explain the data preprocessing pipeline, deduplication, mode imputation, and the mathematical equirectangular planar projection from GPS coordinates $(\phi, \lambda)$ to metric planar coordinates $(x_{\text{km}}, y_{\text{km}})$.
- Explain the statistical sampling methods (SRSWOR, SRSWR, Stratified by zone, Cluster sampling) and equal-width discretization binning.

#### 2. Codebase & File Ownership
- [src/grievanceroute/data/generator.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/data/generator.py) *(8,040 synthetic records generator, Hinglish templates, defects)*
- [src/grievanceroute/data/preprocessor.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/data/preprocessor.py) *(Cleaning pipeline, coordinate projection, leakage prevention)*
- [src/grievanceroute/data/validator.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/data/validator.py) *(Schema validation, geographic bounding box)*
- [scripts/generate_dataset.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/scripts/generate_dataset.py) *(Dataset generation script)*
- [scripts/preprocess_data.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/scripts/preprocess_data.py) *(Data cleaning execution)*
- [scripts/validate_dataset.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/scripts/validate_dataset.py) *(Data integrity validation)*
- [pages/07_Data_Quality.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/07_Data_Quality.py) *(Unit III Lab page: cleaning, scaling, sampling, binning)*
- [tests/test_data_pipeline.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/tests/test_data_pipeline.py) *(Pytest tests for reproducibility, leakage, and projection)*

#### 3. Slide Presentation Breakdown (Slides 1 – 6)
- **Slide 1**: Title, Group 37 introduction, Project Motivation & Civic Challenges in Pune.
- **Slide 2**: Literature Survey Summary & Identified Research Gaps.
- **Slide 3**: End-to-End System Block Diagram & Multi-Tier Architecture.
- **Slide 4**: Dataset Architecture (18 features, 8,040 raw records, Hinglish & formal phrasing).
- **Slide 5**: Injected Defects & Preprocessing Pipeline (Deduplication, Imputation, Boundary filtering).
- **Slide 6**: Planar Metric Projection Mathematics & Unit III Sampling Demos.

#### 4. Live Demonstration & Viva Defense Role
- **Live Demo**: Open **Page 07 (Data Quality & Insights)**. Demonstrate null imputation comparison, Min-Max vs. Standard Z-score scaling, toggle between SRSWOR and Stratified sampling, and showcase the equal-width binning histogram.
- **Viva Defense Questions to Answer**:
  - *Why can't we use raw latitude/longitude degrees directly in K-Means or Euclidean distance?*
  - *How does the system ensure zero target leakage during preprocessing?*
  - *What is the difference between SRSWOR and Stratified Sampling in terms of variance?*

---

### 👤 Person 2: NLP & Supervised Machine Learning Engineer
*“Automating Complaint Triage, Multi-Target Regression, and Boundary Error Analysis”*

#### 1. Core Responsibilities
- Explain the NLP feature extraction pipeline: sublinear TF-IDF vectorization across word unigrams and bigrams (`ngram_range=(1, 2)`).
- Explain the multiclass text classification models: Multinomial Logistic Regression (primary), Multinomial Naive Bayes, Decision Trees, and evaluation against the **Majority Class Dummy Baseline**.
- Present the multi-target regression models for continuous Urgency Score (1–5) and Actual Resolution Duration (hours) evaluated against the **Median Baseline**.
- Explain the **Error Analysis**: Discuss the 3.6% realistic misclassification rate, why cross-domain grievances (e.g. water leaks causing asphalt craters) confound bag-of-words models, and how the confusion matrix reflects genuine municipal ambiguity.
- Discuss feature explainability: Top discriminative TF-IDF tokens per category extracted from model weights.

#### 2. Codebase & File Ownership
- [src/grievanceroute/ml/classifier.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/ml/classifier.py) *(TF-IDF pipelines, Logistic Regression, Naive Bayes, Decision Tree, Majority baseline, sample error logging)*
- [src/grievanceroute/ml/regression.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/ml/regression.py) *(Linear, Ridge, Random Forest regressors, Median baseline, residual analysis)*
- [scripts/train_models.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/scripts/train_models.py) *(Model training, evaluation, and artifact serialization)*
- [pages/04_ML_Workbench.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/04_ML_Workbench.py) *(ML Workbench: metrics tables, ROC curves, confusion matrix, error analysis, explainability)*
- [tests/test_ml_models.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/tests/test_ml_models.py) *(Classifier and regressor test assertions)*

#### 3. Slide Presentation Breakdown (Slides 7 – 12)
- **Slide 7**: NLP Feature Pipeline: Sublinear TF-IDF Unigrams + Bigrams.
- **Slide 8**: Text Classification Benchmark: Logistic Regression (96.37%) vs. Majority Baseline (26.87%).
- **Slide 9**: Confusion Matrix & One-vs-Rest Multiclass ROC Curves ($AUC \approx 0.99$).
- **Slide 10**: Multi-Target Regression: Random Forest vs. Median Baseline for Urgency ($R^2=0.21$) & Resolution Hours ($R^2=0.17$).
- **Slide 11**: Error Analysis & Boundary Ambiguity (Cross-domain citizen narratives and secondary topic dominance).
- **Slide 12**: NLP Explainability: Top Discriminative Token Weights per Category.

#### 4. Live Demonstration & Viva Defense Role
- **Live Demo**: Open **Page 04 (ML Workbench)**. Switch between tabs: Model Comparison, Confusion Matrix & ROC curves, **Representative Misclassification Error Analysis**, and Feature Explainability.
- **Viva Defense Questions to Answer**:
  - *Why does Logistic Regression perform so well on high-dimensional TF-IDF sparse vectors?*
  - *Why is evaluating against a Median baseline critical for regression credibility?*
  - *What causes a complaint about road damage from a broken pipeline to be misclassified as a water leak?*

---

### 👤 Person 3: Spatial Intelligence & Analytics Engineer
*“Spatial Clustering, Density Hotspot Mining, and Municipal Operations Analytics”*

#### 1. Core Responsibilities
- Explain the formulation of spatial clustering on projected Cartesian coordinates $(x_{\text{km}}, y_{\text{km}})$ rather than latitude/longitude.
- Explain K-Means spatial hotspot partitioning: Silhouette analysis ($S=0.4864$ at $K=5$) and Inertia ($46,768.5$) across Pune municipal zones.
- Explain DBSCAN density-based hotspot detection: $\epsilon$-neighborhood parameterization ($\epsilon = 0.8\text{ km}$), core sample threshold ($\text{min\_samples} = 25$), and the operational role of noise point filtering.
- Explain Apriori association rule mining from scratch: Transaction discretization, Support, Confidence, Lift, and how rules are mined strictly on environmental variables (`zone`, `weather`, `time_of_day`, `monsoon`, `low_lying`) to avoid target leakage.
- Present the real-time Municipal Operations Dashboard KPIs.

#### 2. Codebase & File Ownership
- [src/grievanceroute/ml/clustering.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/ml/clustering.py) *(Projected K-Means, DBSCAN density clustering, centroid extraction)*
- [src/grievanceroute/ml/association_rules.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/ml/association_rules.py) *(From-scratch Apriori algorithm, candidate generation, lift evaluation)*
- [pages/01_Operations_Dashboard.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/01_Operations_Dashboard.py) *(Real-time operational KPIs, urgent queue, category & zone charts)*
- [pages/03_Geographic_Hotspots.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/03_Geographic_Hotspots.py) *(Interactive Plotly 7+ K-Means & DBSCAN maps, centroid summaries)*
- [scripts/run_experiments.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/scripts/run_experiments.py) *(Spatial clustering grids, DBSCAN sweeps, and Apriori mining)*
- [scripts/generate_report_assets.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/scripts/generate_report_assets.py) *(Folium cluster maps, figures, and plots)*

#### 3. Slide Presentation Breakdown (Slides 13 – 17)
- **Slide 13**: Municipal Spatial Analytics Problem & Coordinate Systems.
- **Slide 14**: K-Means Hotspot Partitioning: Silhouette Optimization ($K=3\dots8$) & Centroid Workload.
- **Slide 15**: DBSCAN Density Clustering: Corridors vs. Isolated Spatial Noise Points.
- **Slide 16**: Apriori Association Rule Mining: Mining Contextual Hazards Without Target Leakage.
- **Slide 17**: Real-Time Operations Dashboard & Priority Incident Dispatch Queue.

#### 4. Live Demonstration & Viva Defense Role
- **Live Demo**: Open **Page 01 (Operations Dashboard)** to showcase high-urgency priority queues. Then open **Page 03 (Geographic Hotspots)**: adjust the $K$ slider (3 to 8) to show dynamic centroid updates, toggle to DBSCAN to demonstrate noise point filtering, and display the interactive raw category incident map.
- **Viva Defense Questions to Answer**:
  - *How does DBSCAN determine whether an incident point is a core, border, or noise point?*
  - *What does a Lift value greater than 1 signify in Apriori association rules?*
  - *How do K-Means centroids assist municipal supervisors in establishing field depot bases?*

---

### 👤 Person 4: Algorithms, Routing & Systems Engineer
*“Combinatorial Heuristic Search, Multi-Crew VRP Dispatch, Safety Rules & MLOps”*

#### 1. Core Responsibilities
- Explain the Pune municipal road network graph model (16 nodes, 36 directed edges) and the shortest-path graph search.
- Explain $A^*$ graph search: Node evaluation function $f(n) = g(n) + h(n)$ with admissible Euclidean distance lower bound ($h(n) \le h^*(n)$).
- Formulate the Multi-Crew Vehicle Routing Problem (VRP) objective function:
  $$J(R) = \alpha \cdot D(R) + \beta \cdot U_{\text{missed}}(R) + \gamma \cdot T_{\text{overtime}}(R)$$
- Explain the combinatorial tour optimization heuristics: 2-Opt Hill Climbing and Simulated Annealing with exponential cooling ($T_k = T_0 \cdot 0.92^k$).
- Present the route feasibility results: Over 27% distance reduction while strictly guaranteeing **0.00 hours overtime** under hard 8.0-hour shift budgets.
- Explain the deterministic safety-critical rule overrides (`"live wire"` $\rightarrow$ Urgency 5, Electrical Crew).
- Present the MLOps engineering, SQLite schema & persistence, and the 21 automated pytest integration tests.

#### 2. Codebase & File Ownership
- [src/grievanceroute/ai/search.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/ai/search.py) *($A^*$ search with admissible Euclidean lower bound)*
- [src/grievanceroute/ai/local_search.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/ai/local_search.py) *(2-Opt Hill Climbing & Simulated Annealing tour permutations)*
- [src/grievanceroute/routing/network.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/routing/network.py) *(NetworkX Pune road network graph)*
- [src/grievanceroute/routing/vrp_solver.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/routing/vrp_solver.py) *(Multi-crew dispatch solver, shift constraints, objective $J(R)$)*
- [src/grievanceroute/services/prediction_service.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/services/prediction_service.py) *(NLP inference + safety-critical rule overrides)*
- [src/grievanceroute/services/complaint_service.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/services/complaint_service.py) *(Intake validation and lifecycle transitions)*
- [src/grievanceroute/database/repository.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/src/grievanceroute/database/repository.py) *(SQLite CRUD repository & status history)*
- [app.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/app.py) *(Main portal & Instant Triage card)*
- [pages/02_Complaint_Management.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/02_Complaint_Management.py) *(Citizen complaint intake form & lifecycle tracking)*
- [pages/05_Crew_Route_Planner.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/05_Crew_Route_Planner.py) *(Multi-crew route dispatch, comparative distance charts, Plotly route map)*
- [pages/06_Experiments.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/pages/06_Experiments.py) *(MLflow runs, artifacts registry, CSV exports)*
- [tests/test_ai_algorithms.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/tests/test_ai_algorithms.py) *(A* admissibility, 2-opt, SA tests)*
- [tests/test_routing_and_db.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/tests/test_routing_and_db.py) *(Shift constraints & database tests)*
- [tests/test_end_to_end_workflow.py](file:///c:/Users/bhale/OneDrive/Documents/GrievanceRoute/tests/test_end_to_end_workflow.py) *(Complete end-to-end lifecycle integration tests)*

#### 3. Slide Presentation Breakdown (Slides 18 – 24)
- **Slide 18**: Multi-Crew Vehicle Routing Problem (VRP) Formulation & Objective Function $J(R)$.
- **Slide 19**: $A^*$ Shortest-Path Search & Admissible Euclidean Lower Bound Heuristic.
- **Slide 20**: Combinatorial Heuristic Solvers: Greedy vs. 2-Opt Hill Climbing vs. Simulated Annealing.
- **Slide 21**: Route Optimization Benchmark: 28% Distance Reduction with 0.00 Overtime Hours.
- **Slide 22**: Deterministic Safety-Critical Overrides & Production System Integration.
- **Slide 23**: System Persistence Architecture (SQLite Schema, Audit History) & MLOps CI Testing (21 Pytest tests).
- **Slide 24**: Project Conclusion, Future Scope & Summary of Engineering Outcomes.

#### 4. Live Demonstration & Viva Defense Role
- **Live Demo**:
  1. Open **Main Portal (`app.py`)**: Type a hazardous complaint (*"Dangerous sparking live wire hanging near FC Road"*). Show how the instant triage triggers an **emergency safety override** (Category $\rightarrow$ `streetlight`, Urgency $\rightarrow$ 5).
  2. Open **Page 02 (Complaint Management)**: Submit a validated complaint; show atomic SQLite insertion and status update (`Pending` $\rightarrow$ `Assigned`).
  3. Open **Page 05 (Crew Route Planner)**: Run the VRP dispatch solver; display comparative distance reduction bar charts and the interactive Plotly crew tour map.
- **Viva Defense Questions to Answer**:
  - *Why does straight-line Euclidean distance guarantee admissibility in $A^*$ search on a planar road graph?*
  - *How does Simulated Annealing avoid getting trapped in suboptimal local minima during tour refinement?*
  - *How does the VRP scheduler guarantee that crews never violate the 8-hour shift constraint?*

---

## ⏱️ Recommended Presentation Time Allocation

| Segment | Presenter | Duration | Topic Focus |
| :---: | :---: | :---: | :--- |
| **Part 1** | **Person 1** | 4 mins | Problem Statement, Motivation, Data Engineering, Injected Defects & Preprocessing |
| **Part 2** | **Person 2** | 4 mins | TF-IDF NLP Triage, Multi-Target Regression, Baselines & Error Analysis |
| **Part 3** | **Person 3** | 4 mins | Projected K-Means, DBSCAN Hotspot Detection, Apriori Mining & Operations Dashboard |
| **Part 4** | **Person 4** | 4 mins | $A^*$ Search, Multi-Crew VRP Dispatch, Simulated Annealing, Safety Overrides & MLOps |
| **Live Demo**| **All 4 Team Members** | 4 mins | Interactive walkthrough of the 7 Streamlit pages matching each person's module |
| **Q&A / Viva**| **All 4 Team Members** | 5 mins | Defense of mathematical principles, code implementation, and design decisions |

---

## 🎯 Quick Reference Cheat Sheet for Examiners / Evaluators

| If Examiner Asks About... | Designated Respondent | Key Points to Cite |
| :--- | :---: | :--- |
| **Data Generation, Defects, Sampling** | **Person 1** | 8,040 raw records, 40 duplicates removed, 170 weather nulls imputed with mode, equirectangular planar projection $(x, y)_{\text{km}}$. |
| **Classification Accuracy, F1, Overfitting** | **Person 2** | Logistic Regression achieved 96.37% accuracy vs. 26.87% Majority Baseline; 3.6% realistic cross-domain misclassifications explained in Error Analysis tab. |
| **Regression Performance & R² Scores** | **Person 2** | Random Forest achieved MAE 0.7656 on Urgency ($R^2=0.21$) and MAE 11.00h on Resolution ($R^2=0.17$), evaluated against naive Median Baselines. |
| **Spatial Coordinates & Clustering** | **Person 3** | K-Means ($K=5$, Silhouette 0.4864) and DBSCAN ($\epsilon=0.8\text{ km}$, 2.91% noise) run on projected metric planar kilometers, avoiding lat/lon distortion. |
| **Apriori Mining & Target Leakage** | **Person 3** | 834 rules extracted from discretized environmental variables (`zone`, `weather`, `monsoon`); post-intake targets excluded to guarantee zero leakage. |
| **VRP Solvers, Shift Budgets, Overtime** | **Person 4** | Simulated Annealing and 2-Opt achieved 28% distance reduction; 8-hour shift limits strictly enforced with 0.00 overtime hours across 8 crews. |
| **Safety Overrides & Emergency Rules** | **Person 4** | Substring rules intercept acute electrical/structural hazards (`"live wire"`, `"sparking"`) to override statistical model predictions for public safety. |
| **Database ACID, MLOps, Automated Tests** | **Person 4** | SQLite with parameterized queries and foreign keys; 21 automated pytest integration tests passing in CI in 5.25 seconds. |
