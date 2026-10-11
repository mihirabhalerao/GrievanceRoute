# GrievanceRoute — Mathematical Algorithm Specifications

## 1. Syllabus & Laboratory Coverage

| Syllabus Unit | Algorithm / Method | Implementation Module | Application Context |
| :--- | :--- | :--- | :--- |
| **Unit I: Intro to AI** | Production Rules, State Space | `grievanceroute.services.prediction_service` | Safety escalation, triage validation |
| **Unit II: Heuristic Search** | A* Search ($f = g + h$) | `grievanceroute.ai.search` | Road network shortest paths |
| | Greedy Best-First ($f = h$) | `grievanceroute.ai.search` | Baseline pathfinding |
| | Hill Climbing (2-opt) | `grievanceroute.ai.local_search` | Route sequence refinement |
| | Simulated Annealing | `grievanceroute.ai.local_search` | Tour optimization |
| | AO* (AND-OR Search) | `grievanceroute.ai.game_and_or` | Problem decomposition |
| | Minimax + $\alpha$-$\beta$ Pruning | `grievanceroute.ai.game_and_or` | Adversarial game trees |
| | Water-Jug (BFS) | `grievanceroute.ai.classic` | State space problem |
| | 8-Puzzle (Manhattan A*) | `grievanceroute.ai.classic` | Sliding tile benchmark |
| **Unit III: Preprocessing** | TF-IDF, Standard/MinMax Scaling, Binning | `grievanceroute.data.preprocessor` | Clean complaints, leakage prevention |
| **Unit IV: Supervised Learning** | Logistic Regression, Decision Tree, KNN, Naive Bayes | `grievanceroute.ml.classifier` | Multiclass complaint triage |
| | Linear & Random Forest Regressors | `grievanceroute.ml.regression` | Urgency & resolution duration |
| **Unit V: Unsupervised Learning** | K-Means (Projected Coordinates) | `grievanceroute.ml.clustering` | Spatial complaint hotspots |
| | DBSCAN Density Clustering | `grievanceroute.ml.clustering` | Dense clusters & spatial noise |
| | Apriori Association Rules | `grievanceroute.ml.association_rules` | Contextual co-occurrence mining |
| **Unit VI: Optimization** | SGD, Momentum, RMSprop, Adam | `grievanceroute.ml.neural_net` | Surrogate loss optimization |

---

## 2. Formulation Details

### A. A* Spatial Graph Search
A* evaluates the node evaluation function:
$$f(n) = g(n) + h(n)$$
- $g(n)$: Accumulated travel cost (km) from start to node $n$.
- $h(n)$: Admissible straight-line Euclidean distance lower bound from node $n$ to goal $g$:
$$h(n) = \sqrt{\Delta y_{\text{km}}^2 + \Delta x_{\text{km}}^2}$$
Because $h(n)$ is Euclidean distance across planar coordinates, $h(n) \le c(n, g)$ in all cases, guaranteeing admissibility and optimality.

### B. Multi-Crew Vehicle Routing Objective
The dispatch scheduler optimizes global objective:
$$J(R) = \alpha \cdot D(R) + \beta \cdot U_{\text{missed}}(R) + \gamma \cdot T_{\text{overtime}}(R)$$
Where:
- $D(R) = \sum_{k \in \text{Crews}} \text{TourDistance}(R_k)$
- $U_{\text{missed}}(R) = \sum_{c \in \text{Unserved}} \text{UrgencyScore}(c)$
- $T_{\text{overtime}}(R) = \sum_{k \in \text{Crews}} \max(0, \text{TotalHours}(R_k) - 8.0)$
- Default weights: $\alpha = 1.0, \beta = 10.0, \gamma = 5.0$.

### C. Simulated Annealing Cooling Schedule
At iteration $k$ with temperature $T_k = T_0 \cdot \gamma^k$ (where $\gamma = 0.92$):
$$\Delta E = J(R_{\text{neighbor}}) - J(R_{\text{current}})$$
- If $\Delta E < 0$, accept move unconditionally.
- If $\Delta E \ge 0$, accept with Boltzmann probability:
$$P(\text{accept}) = \exp\left(-\frac{\Delta E}{T_k}\right)$$

### D. Apriori Association Rule Metrics
- **Support**:
$$\text{Support}(A \Rightarrow B) = P(A \cup B) = \frac{\text{Count}(A \cup B)}{N}$$
- **Confidence**:
$$\text{Confidence}(A \Rightarrow B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
- **Lift**:
$$\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)}$$
*Note:* Association does not imply causation. Synthetic association rules demonstrate algorithm mechanics rather than empirical causal phenomena.
