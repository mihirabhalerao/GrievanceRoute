# GrievanceRoute — College Mini-Project Report Mapping & Outline

Aligned with the Third Year B.Tech (Computer Engineering) AIML Laboratory Mini-Project Report Format (MKSSS’s Cummins College of Engineering for Women, Pune).

---

## Section 1: Introduction
- **Civic Problem**: High-volume, heterogeneous citizen grievances in urban governance (potholes, water leaks, uncollected refuse, electrical hazards).
- **Manual Triage Bottleneck**: Slow manual operator categorization and lack of immediate safety prioritization.
- **Proposed System**: *GrievanceRoute* — An intelligent civic operations platform integrating NLP triage, multi-target urgency regression, spatial density analysis, and vehicle route optimization.

## Section 2: Motivation
- Analogy to emergency hospital triage nurse.
- Public safety risks from delayed electrical or flooding emergencies.
- Need for transparent safety escalation overrides alongside probabilistic ML predictions.

## Section 3: Literature Survey
Comprehensive review of 5 verified literature citations:
1. **T. Joachims (1998)**: *Text categorization with support vector machines*. High-dimensional sparse TF-IDF text features.
2. **K. Kowsari et al. (2019)**: *Text classification algorithms: A survey*. Structured comparison of linear vs deep classifiers.
3. **X. Li et al. (2025)**: *Intelligent system for classifying patient complaints using machine learning and NLP*. Multiclass complaint triage.
4. **M. Z. Al Amin et al. (2025)**: *From text to action: AI-driven classification of public service complaints*. Stratified TF-IDF on municipal complaints.
5. **P. Sabilla and R. Dhenabayu (2026)**: *Comparative complaint text classification using SVM, CNN, and BERT with priority scoring for NYC 311 service requests*. Priority score integration.

## Section 4: Dataset Used
- **Dataset Description**: Synthetic Pune Civic Complaints (`data/raw/pune_civic_complaints.csv`, 8,040 rows raw, 8,000 cleaned).
- **Features**: 18 attributes (NLP text, coordinates, weather, recurrence, urgency, resolution time).
- **Defects Injected**: 40 duplicate records, 104 null weather values for preprocessing exercises.
- **Companion Data**: `crews.csv` (8 crews) and `road_network.json` (16 road nodes).

## Section 5: Flow Diagram / Block Diagram
- System architecture diagram: `artifacts/figures/architecture.png`
- Complaint-to-route workflow: `artifacts/figures/workflow.png`

## Section 6: Methods Used / Technology Stack
- **Language**: Python 3.11 / 3.14
- **Libraries**: Scikit-Learn, Streamlit, Pandas, NumPy, NetworkX, Plotly, Folium, MLflow, Pytest, Joblib.

## Section 7: Model Development
- Preprocessing: whitespace cleaning, mode imputation, deduplication.
- Feature Extraction: Word unigram/bigram TF-IDF vectorizer (sublinear TF).
- Model Exploration: Logistic Regression, Decision Tree, KNN, Multinomial Naive Bayes.
- Regression: Linear Regression, Ridge, Random Forest for urgency (1–5) and resolution hours.

## Section 8: Version Control
- Git repository structure with modular packaging, branch isolation, and GitHub Actions CI.

## Section 9: Experiment Tracking
- MLflow runs tracking model hyperparameter grids, execution runtimes, and validation scores.
- Consolidated run summary: `artifacts/metrics/experiment_summary.csv`.

## Section 10: AutoML / MLOps
- Automated hyperparameter selection via GridSearchCV.
- CI automated testing pipeline: `.github/workflows/tests.yml`.
- Pinned dependency reproduction with Docker.

## Section 11: Deployment
- Multi-page Streamlit web application (`app.py`, `pages/01` to `pages/08`).
- Local-first offline execution on laptop CPU.
- Parameterized SQLite persistence (`data/grievance_route.db`).

## Section 12: Results and Discussions
- **Classification Performance**: Logistic Regression baseline achieves high macro-F1 on template test split (1,600 records).
- **Urgency Regression**: Random Forest achieves MAE 0.79 on 1–5 urgency scale.
- **Resolution Regression**: Random Forest achieves MAE 11.6 hours.
- **Clustering**: K-Means ($K=5$) yields silhouette score 0.545 on projected coordinates.
- **Route Optimization**: Simulated Annealing achieves 15–20% distance reduction over greedy baseline.
- **Caveats**: Synthetic template-based accuracy is inflated; real-world generalizability requires testing on organic citizen submissions.

## Section 13: Future Scope
- Integration with live Pune Municipal Corporation (PMC) citizen portals.
- Marathi NLP support with multilingual regional transformer models (IndicBERT).
- Real-time GPS crew tracking and dynamic vehicle re-routing.

## Section 14: Conclusion
- Summary of end-to-end working system covering all six AIML syllabus units, persistent database, working GUI, and reproducible testing.

## Section 15: References
Complete IEEE-style references verified from publishers and indexing records.
