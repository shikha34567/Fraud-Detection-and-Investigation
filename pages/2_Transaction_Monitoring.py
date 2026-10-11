"""
Page 2: Transaction Monitoring & Search.
Real-time transaction surveillance grid with filtering, searching,
risk score thresholds, detail drawer, and CSV download.
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
from datetime import datetime

from src.database.connection import get_db_session
from src.database.crud import get_transactions_list
from src.utils.ui_components import (
    apply_custom_theme,
    get_risk_badge_html,
    ACCENT_BLUE
)

st.set_page_config(page_title="Transaction Monitoring | FinGuard", page_icon="🔎", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">🔎 Real-Time Transaction Monitoring</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Filter, search, inspect transactions and audit machine learning inference records
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Filters Row
f_col1, f_col2, f_col3, f_col4 = st.columns([2, 1.5, 1.5, 1.5])

with f_col1:
    search_ref = st.text_input("🔍 Search Transaction Reference:", placeholder="e.g. TXN-20261004...")

with f_col2:
    selected_risk = st.selectbox(
        "Risk Category:",
        ["All", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
    )

with f_col3:
    selected_status = st.selectbox(
        "Investigation Status:",
        ["All", "New", "Under Review", "Confirmed Fraud", "Legitimate", "Escalated"]
    )

with f_col4:
    min_score = st.slider("Min Risk Score (0-100):", min_value=0, max_value=100, value=0, step=5)

# Fetch data from DB
with get_db_session() as session:
    records = get_transactions_list(
        session,
        limit=500,
        risk_category=selected_risk if selected_risk != "All" else None,
        min_risk_score=float(min_score),
        investigation_status=selected_status if selected_status != "All" else None,
        search_query=search_ref if search_ref else None
    )

if not records:
    st.info("No transactions found matching the specified filter criteria.")
else:
    df_display = pd.DataFrame(records)
    
    # Download Button & Count
    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        st.write(f"Showing **{len(df_display)}** transaction records")
    with d_col2:
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Export",
            data=csv_data,
            file_name=f"transactions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Interactive Table
    columns_to_show = [
        "reference", "amount", "fraud_probability", "risk_score",
        "risk_category", "anomaly_score", "status", "final_outcome", "created_at"
    ]

    st.dataframe(
        df_display[columns_to_show],
        column_config={
            "reference": st.column_config.TextColumn("Reference ID"),
            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
            "fraud_probability": st.column_config.NumberColumn("Probability", format="%.4f"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", format="%.1f", min_value=0, max_value=100),
            "risk_category": st.column_config.TextColumn("Risk Tier"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly Score", format="%.3f"),
            "status": st.column_config.TextColumn("Investigation Status"),
            "final_outcome": st.column_config.TextColumn("Outcome"),
            "created_at": st.column_config.TextColumn("Timestamp")
        },
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # Transaction Detail Inspector
    st.subheader("📋 Inspect Transaction Payload")
    selected_ref = st.selectbox(
        "Select Reference to view underlying features and payload:",
        df_display["reference"].tolist()
    )

    if selected_ref:
        selected_record = next(r for r in records if r["reference"] == selected_ref)
        
        det1, det2, det3 = st.columns([1, 1, 1])
        with det1:
            st.markdown(f"**Reference:** `{selected_record['reference']}`")
            st.markdown(f"**Amount:** `${selected_record['amount']:.2f}`")
            st.markdown(f"**Time Offset:** `{selected_record['time']}` sec")
        with det2:
            st.markdown(f"**Probability:** `{selected_record['fraud_probability']:.4f}`")
            st.markdown(f"**Risk Category:** {get_risk_badge_html(selected_record['risk_category'])}", unsafe_allow_html=True)
            st.markdown(f"**Anomaly Score:** `{selected_record['anomaly_score']:.4f}`")
        with det3:
            st.markdown(f"**Investigation Status:** `{selected_record['status']}`")
            st.markdown(f"**Outcome:** `{selected_record['final_outcome']}`")
            st.markdown(f"**Logged At:** `{selected_record['created_at']}`")

        with st.expander("🔍 View Raw PCA Features (V1 - V28)"):
            st.json(selected_record["features"])
