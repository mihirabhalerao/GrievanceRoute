"""
UI Theme and Styling Helpers for GrievanceRoute Streamlit Application.
Provides rich, modern styling, metric cards, badges, and consistent branding.
"""

import streamlit as st


def apply_custom_theme():
    """Inject custom modern CSS styling into the Streamlit app."""
    st.markdown("""
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header hero styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 14px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }
    
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        flex: 1;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        border-left: 4px solid #3b82f6;
    }
    
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* Badges */
    .badge-urgent {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-resolved {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    
    .badge-pending {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Card panel */
    .content-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, badge: str = "Pune Civic AI"):
    """Render a modern hero header banner."""
    st.markdown(f"""
    <div class="hero-banner">
        <span style="background: rgba(255,255,255,0.15); padding: 3px 10px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: #38bdf8;">
            {badge}
        </span>
        <h1 class="hero-title" style="margin-top: 10px;">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
