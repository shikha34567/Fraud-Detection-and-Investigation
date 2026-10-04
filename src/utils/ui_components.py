"""
Reusable Streamlit UI Components, Design System, and Custom Styling.
Provides executive dark-navy financial theme, risk badges, metrics cards, and Plotly styling.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List


# Color Tokens
DARK_BG = "#0B0F19"
CARD_BG = "#131B2E"
BORDER_COLOR = "#22304A"
ACCENT_BLUE = "#38BDF8"
ACCENT_CYAN = "#06B6D4"
SUCCESS_GREEN = "#10B981"
WARNING_AMBER = "#F59E0B"
DANGER_RED = "#EF4444"
CRITICAL_PURPLE = "#DC2626"
TEXT_PRIMARY = "#F8FAFC"
TEXT_MUTED = "#94A3B8"


def apply_custom_theme():
    """Injects financial risk dashboard styling into Streamlit DOM."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            /* Base Background */
            .stApp {{
                background-color: {DARK_BG};
                color: {TEXT_PRIMARY};
            }}

            /* Sidebar styling */
            section[data-testid="stSidebar"] {{
                background-color: #0d1424;
                border-right: 1px solid {BORDER_COLOR};
            }}

            /* Cards */
            .metric-card {{
                background: linear-gradient(135deg, {CARD_BG} 0%, #17233d 100%);
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
                padding: 20px 24px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }}
            .metric-card:hover {{
                border-color: {ACCENT_BLUE};
                transform: translateY(-2px);
            }}

            .metric-title {{
                font-size: 0.85rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: {TEXT_MUTED};
                margin-bottom: 8px;
            }}

            .metric-value {{
                font-size: 2.1rem;
                font-weight: 700;
                color: {TEXT_PRIMARY};
                letter-spacing: -0.02em;
            }}

            .metric-sub {{
                font-size: 0.8rem;
                color: {TEXT_MUTED};
                margin-top: 6px;
            }}

            /* Risk Badges */
            .risk-badge {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                letter-spacing: 0.03em;
            }}
            .badge-low {{
                background-color: rgba(16, 185, 129, 0.15);
                color: {SUCCESS_GREEN};
                border: 1px solid rgba(16, 185, 129, 0.3);
            }}
            .badge-medium {{
                background-color: rgba(245, 158, 11, 0.15);
                color: {WARNING_AMBER};
                border: 1px solid rgba(245, 158, 11, 0.3);
            }}
            .badge-high {{
                background-color: rgba(239, 68, 68, 0.15);
                color: {DANGER_RED};
                border: 1px solid rgba(239, 68, 68, 0.3);
            }}
            .badge-critical {{
                background-color: rgba(220, 38, 38, 0.25);
                color: #ff4d4d;
                border: 1px solid #ff4d4d;
                animation: pulse 2s infinite;
            }}

            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }}
                70% {{ box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }}
            }}

            /* Status Pills */
            .status-pill {{
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            .status-new {{ background: #2563eb22; color: #60a5fa; border: 1px solid #2563eb66; }}
            .status-review {{ background: #d9770622; color: #fbbf24; border: 1px solid #d9770666; }}
            .status-fraud {{ background: #dc262622; color: #f87171; border: 1px solid #dc262666; }}
            .status-legit {{ background: #05966922; color: #34d399; border: 1px solid #05966966; }}
            .status-escalated {{ background: #7c3aed22; color: #c084fc; border: 1px solid #7c3aed66; }}

            /* Table customizations */
            div[data-testid="stDataFrame"] {{
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid {BORDER_COLOR};
            }}

            /* Header banner */
            .header-banner {{
                background: linear-gradient(90deg, #131B2E 0%, #1e293b 100%);
                border-bottom: 1px solid {BORDER_COLOR};
                padding: 18px 24px;
                border-radius: 12px;
                margin-bottom: 24px;
            }}

            /* Notice / Alert box */
            .system-notice {{
                background-color: rgba(56, 189, 248, 0.08);
                border-left: 4px solid {ACCENT_BLUE};
                padding: 12px 18px;
                border-radius: 0 8px 8px 0;
                font-size: 0.88rem;
                color: #e2e8f0;
                margin-bottom: 18px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_metric_card(title: str, value: str, subtext: str = "", delta_color: str = "normal"):
    """Renders a styled dark executive metric card."""
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def get_risk_badge_html(category: str, score: float | None = None) -> str:
    """Returns HTML markup for risk badges with appropriate colors."""
    cat_lower = category.lower()
    score_str = f" ({score:.1f})" if score is not None else ""
    
    if "critical" in cat_lower:
        cls = "badge-critical"
    elif "high" in cat_lower:
        cls = "badge-high"
    elif "medium" in cat_lower:
        cls = "badge-medium"
    else:
        cls = "badge-low"

    return f'<span class="risk-badge {cls}">{category}{score_str}</span>'


def get_plotly_layout_defaults() -> Dict[str, Any]:
    """Returns standardized dark Plotly layout config matching dashboard theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 27, 46, 0.4)",
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=12),
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED)
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_MUTED)
        ),
        legend=dict(
            bgcolor="rgba(19, 27, 46, 0.8)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_PRIMARY)
        )
    )
