"""
Page 1: Executive Risk & Fraud Dashboard.
Presents high-level metrics, volume over time, risk tier breakdowns,
fraud probability distributions, and high-priority suspicious alerts.
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

from src.database.connection import get_db_session
from src.database.crud import get_dashboard_summary_metrics, get_recent_alerts, get_transactions_list
from src.utils.ui_components import (
    apply_custom_theme,
    render_metric_card,
    get_risk_badge_html,
    get_plotly_layout_defaults,
    ACCENT_BLUE,
    SUCCESS_GREEN,
    WARNING_AMBER,
    DANGER_RED,
    CARD_BG
)

st.set_page_config(page_title="Executive Dashboard | FinGuard", page_icon="📊", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">📊 Executive Risk Dashboard</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Macro-level transaction surveillance, risk distributions, and priority queue
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with get_db_session() as session:
    metrics = get_dashboard_summary_metrics(session)
    recent_alerts = get_recent_alerts(session, limit=10)
    txns = get_transactions_list(session, limit=250)

df_txns = pd.DataFrame(txns)

# Top KPIs Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    render_metric_card("Total Analyzed", f"{metrics['total_transactions']:,}", "All processed transactions")
with kpi2:
    render_metric_card("Flagged Alerts", f"{metrics['flagged_transactions']:,}", "High & Critical Risk cases")
with kpi3:
    render_metric_card("Predicted Fraud", f"{metrics['predicted_fraud_count']:,}", "Model score p >= 0.50")
with kpi4:
    rate = (
        (metrics['predicted_fraud_count'] / metrics['total_transactions'] * 100)
        if metrics['total_transactions'] > 0 else 0.0
    )
    render_metric_card("Predicted Fraud Rate", f"{rate:.2f}%", "Proportion of volume flagged")

st.markdown("<br>", unsafe_allow_html=True)

if df_txns.empty:
    st.info("💡 No transactions recorded yet. Go to **6_Transaction_Simulator** to stream transactions into the database!")
else:
    # Row 1 Charts: Risk Tier Distribution & Volume Over Time
    c1, c2 = st.columns(2)
    layout_def = get_plotly_layout_defaults()

    with c1:
        st.subheader("Risk Category Distribution")
        cat_counts = df_txns["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["Risk Category", "Count"]

        color_map = {
            "Low Risk": SUCCESS_GREEN,
            "Medium Risk": WARNING_AMBER,
            "High Risk": DANGER_RED,
            "Critical Risk": "#DC2626"
        }

        fig_pie = px.pie(
            cat_counts,
            names="Risk Category",
            values="Count",
            color="Risk Category",
            color_discrete_map=color_map,
            hole=0.45
        )
        fig_pie.update_layout(layout_def)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Transaction Volume & Cumulative Risk")
        # Bin time or sequence
        df_sorted = df_txns.sort_values(by="id")
        fig_vol = px.scatter(
            df_sorted,
            x="id",
            y="amount",
            color="risk_category",
            size="risk_score",
            color_discrete_map=color_map,
            hover_data=["reference", "risk_score", "fraud_probability"],
            labels={"id": "Transaction Sequence", "amount": "Amount ($)"}
        )
        fig_vol.update_layout(layout_def)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Row 2 Charts: Fraud Probability Distribution & Amount Trends
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Fraud Probability Distribution")
        fig_hist = px.histogram(
            df_txns,
            x="fraud_probability",
            nbins=30,
            color="risk_category",
            color_discrete_map=color_map,
            labels={"fraud_probability": "Predicted Fraud Probability"}
        )
        fig_hist.update_layout(layout_def)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        st.subheader("Transaction Amount by Risk Level")
        fig_box = px.box(
            df_txns,
            x="risk_category",
            y="amount",
            color="risk_category",
            color_discrete_map=color_map,
            points="all",
            labels={"amount": "Amount ($)", "risk_category": "Risk Category"}
        )
        fig_box.update_layout(layout_def)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Priority Suspicious Alert Queue
    st.subheader("🚨 Priority Suspicious Alert Feed (High & Critical Risk)")
    if recent_alerts:
        df_alerts = pd.DataFrame(recent_alerts)
        st.dataframe(
            df_alerts,
            column_config={
                "reference": st.column_config.TextColumn("Transaction Reference"),
                "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.1f", min_value=0, max_value=100),
                "fraud_probability": st.column_config.NumberColumn("Fraud Probability", format="%.4f"),
                "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "risk_category": st.column_config.TextColumn("Category"),
                "status": st.column_config.TextColumn("Investigation Status"),
                "created_at": st.column_config.TextColumn("Timestamp")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.success("✅ No uninvestigated High or Critical Risk transactions detected.")
