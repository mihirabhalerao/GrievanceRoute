"""
Report Assets and Diagrams Generator for GrievanceRoute.
Generates architecture.png, workflow.png, and cluster_map.html.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import folium


def main():
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)

    # 1. Architecture diagram
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    layers = [
        ("UI / Presentation Layer\n(Streamlit Multipage: Dashboard, Triage, Hotspots, Route Planner, AI Lab)", 5.5, "#3b82f6"),
        ("Service & Orchestration Layer\n(Complaint Service, Prediction Service, Planning Service)", 4.0, "#10b981"),
        ("Core AI & ML Engine\n(TF-IDF + LR, Regressors, K-Means/DBSCAN, Apriori, A*, 2-Opt / SA, AO*, Minimax)", 2.5, "#8b5cf6"),
        ("Data & Persistence Layer\n(SQLite Database, Pandas Pipelines, Joblib Model Registry, NetworkX Graph)", 1.0, "#f59e0b"),
    ]

    for label, y, color in layers:
        rect = patches.FancyBboxPatch((1, y - 0.5), 8, 0.9, boxstyle="round,pad=0.2", facecolor=color, alpha=0.85, edgecolor="none")
        ax.add_patch(rect)
        ax.text(5, y, label, ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    plt.title("GrievanceRoute — End-to-End System Architecture", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("artifacts/figures/architecture.png")
    plt.close()

    # 2. Workflow diagram
    fig, ax = plt.subplots(figsize=(11, 4), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    steps = [
        ("Citizen\nIntake", 1.0, "#0284c7"),
        ("NLP Triage &\nSafety Override", 3.2, "#0ea5e9"),
        ("Urgency &\nResolution ML", 5.4, "#06b6d4"),
        ("Hotspots &\nGrouping", 7.6, "#14b8a6"),
        ("Multi-Crew\nVRP Dispatch", 9.8, "#10b981"),
    ]

    for i, (text, x, color) in enumerate(steps):
        rect = patches.FancyBboxPatch((x - 0.85, 1.2), 1.7, 1.6, boxstyle="round,pad=0.15", facecolor=color, edgecolor="none")
        ax.add_patch(rect)
        ax.text(x, 2.0, text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i + 1][1] - 0.95, 2.0), xytext=(x + 0.95, 2.0),
                        arrowprops=dict(arrowstyle="->", lw=2.5, color="#475569"))

    plt.title("GrievanceRoute — Civic Grievance Triage & Route Dispatch Workflow", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("artifacts/figures/workflow.png")
    plt.close()

    # 3. Folium Cluster Map
    pune_map = folium.Map(location=[18.5204, 73.8567], zoom_start=12, tiles="CartoDB positron")
    cleaned_csv = Path("data/processed/cleaned_complaints.csv")
    if cleaned_csv.exists():
        df = pd.read_csv(cleaned_csv).head(150)
        for _, r in df.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5,
                popup=f"{r['complaint_id']}: {r['complaint_type']} (Urgency {r['urgency_score']})",
                color="#dc2626" if r["urgency_score"] >= 4 else "#2563eb",
                fill=True,
                fill_opacity=0.7,
            ).add_to(pune_map)
    pune_map.save("artifacts/figures/cluster_map.html")
    print("Report assets generated successfully.")


if __name__ == "__main__":
    main()
