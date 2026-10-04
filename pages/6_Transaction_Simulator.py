"""
Page 6: Real-Time Transaction Simulator.
Simulates incoming streaming credit card transactions, triggers instantaneous ML scoring,
persists records into the SQL database, and displays live visual alerts.
"""

import streamlit as st
import time
import pandas as pd
import plotly.express as px

from src.simulation.transaction_stream import simulator
from src.database.connection import get_db_session
from src.database.crud import get_dashboard_summary_metrics
from src.utils.ui_components import (
    apply_custom_theme,
    render_metric_card,
    get_risk_badge_html,
    get_plotly_layout_defaults,
    ACCENT_BLUE,
    DANGER_RED,
    SUCCESS_GREEN
)

st.set_page_config(page_title="Transaction Simulator | FinGuard", page_icon="⚡", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">⚡ Real-Time Streaming Transaction Simulator</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Simulate real-time payment feed ingestion, automated ML inference, and alert triggering
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="system-notice">
        <strong>⚠️ Simulation Notice:</strong> This streaming engine generates and ingests transactions for demonstration,
        benchmarking, and UI testing. All generated records are strictly labeled as simulated and do not connect to a live payment processor.
    </div>
    """,
    unsafe_allow_html=True
)

# Simulation State Management
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_processed_count" not in st.session_state:
    st.session_state.sim_processed_count = 0
if "sim_stream_history" not in st.session_state:
    st.session_state.sim_stream_history = []

# Control Bar
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.5, 1.5, 2, 2])

with ctrl_c1:
    batch_size = st.number_input("Batch Size (Txns/Tick):", min_value=1, max_value=20, value=5, step=1)

with ctrl_c2:
    interval_sec = st.slider("Stream Tick Interval (sec):", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

with ctrl_c3:
    st.markdown("<br>", unsafe_allow_html=True)
    b_start, b_pause, b_stop = st.columns(3)
    with b_start:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.sim_running = True
    with b_pause:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.sim_running = False
    with b_stop:
        if st.button("⏹️ Reset", use_container_width=True):
            st.session_state.sim_running = False
            st.session_state.sim_stream_history = []
            st.session_state.sim_processed_count = 0
            st.rerun()

with ctrl_c4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"Status: **{'🟢 STREAMING' if st.session_state.sim_running else '⚪ IDLE / PAUSED'}**")

# Top Metrics Row
with get_db_session() as session:
    metrics = get_dashboard_summary_metrics(session)

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
with m_c1:
    render_metric_card("DB Stored Transactions", f"{metrics['total_transactions']:,}", "All active records")
with m_c2:
    render_metric_card("Session Ingested", f"{st.session_state.sim_processed_count:,}", "Processed this session")
with m_c3:
    render_metric_card("High/Critical Alerts", f"{metrics['flagged_transactions']:,}", "Requires case review")
with m_c4:
    render_metric_card("Active Stream Pool", "ULB / Synthetic", "Simulated transactions")

st.markdown("<br>", unsafe_allow_html=True)

# Run Simulation Tick if active
if st.session_state.sim_running:
    with st.spinner("Streaming transactions..."):
        new_records = simulator.process_and_persist_batch(batch_size=int(batch_size))
        st.session_state.sim_processed_count += len(new_records)
        # Keep latest 50 in memory for stream display
        st.session_state.sim_stream_history = (new_records + st.session_state.sim_stream_history)[:50]

# Display Live Stream Results
st.subheader("🔴 Live Ingestion Stream & Immediate Risk Assessments")

if st.session_state.sim_stream_history:
    df_live = pd.DataFrame(st.session_state.sim_stream_history)
    
    # Stream chart
    fig_stream = px.scatter(
        df_live,
        x=df_live.index,
        y="risk_score",
        size="amount",
        color="risk_category",
        color_discrete_map={
            "Low Risk": SUCCESS_GREEN,
            "Medium Risk": "#F59E0B",
            "High Risk": DANGER_RED,
            "Critical Risk": "#DC2626"
        },
        labels={"index": "Recent Stream Index", "risk_score": "Risk Score (0-100)"},
        title="Live Ingestion Stream: Risk Score vs Ingestion Sequence"
    )
    fig_stream.update_layout(get_plotly_layout_defaults())
    st.plotly_chart(fig_stream, use_container_width=True)

    # Stream table
    st.dataframe(
        df_live[["reference", "amount", "fraud_probability", "risk_score", "risk_category", "anomaly_score"]],
        column_config={
            "reference": st.column_config.TextColumn("Transaction Reference"),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
            "fraud_probability": st.column_config.NumberColumn("Probability", format="%.4f"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.1f", min_value=0, max_value=100),
            "risk_category": st.column_config.TextColumn("Risk Category"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly Score", format="%.3f")
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("Click '▶️ Start' above to stream live simulated transactions into the detection engine.")

# Auto-refresh loop if streaming
if st.session_state.sim_running:
    time.sleep(interval_sec)
    st.rerun()
