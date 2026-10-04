"""
Page 3: Fraud Investigation & Case Management.
In-depth investigative workspace with SHAP attribution explanations,
human-in-the-loop audit logs, investigator notes, and case resolution.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.database.connection import get_db_session
from src.database.crud import (
    get_transactions_list,
    get_transaction_by_reference,
    update_investigation_record
)
from src.explainability.shap_explainer import shap_explainer
from src.utils.ui_components import (
    apply_custom_theme,
    get_risk_badge_html,
    get_plotly_layout_defaults,
    ACCENT_BLUE,
    DANGER_RED,
    SUCCESS_GREEN
)

st.set_page_config(page_title="Fraud Investigation | FinGuard", page_icon="🕵️", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">🕵️ Fraud Investigation & Case Management</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Deep-dive forensic analysis, SHAP risk drivers, case escalation, and resolution audit trail
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Fetch potential cases
with get_db_session() as session:
    all_txns = get_transactions_list(session, limit=300)

if not all_txns:
    st.info("No transactions available to investigate. Generate simulated transactions in **6_Transaction_Simulator**.")
    st.stop()

# Case Selection
ref_options = [
    f"{t['reference']} | Score: {t['risk_score']:.1f} | {t['risk_category']} | Status: {t['status']}"
    for t in all_txns
]

selected_option = st.selectbox("Select Case to Investigate:", ref_options, index=0)
selected_ref = selected_option.split(" | ")[0]

with get_db_session() as session:
    case = get_transaction_by_reference(session, selected_ref)

if not case:
    st.error("Case could not be found.")
    st.stop()

# Transaction Overview Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"**Transaction Ref:**<br>`{case['reference']}`", unsafe_allow_html=True)
    st.markdown(f"**Amount:** `${case['amount']:.2f}`")
with c2:
    st.markdown(f"**Fraud Probability:** `{case['fraud_probability']:.4f}`")
    st.markdown(f"**Risk Score:** `{case['risk_score']:.1f} / 100`")
with c3:
    st.markdown(f"**Risk Category:**<br>{get_risk_badge_html(case['risk_category'])}", unsafe_allow_html=True)
    st.markdown(f"**Anomaly Score:** `{case['anomaly_score']:.4f}`")
with c4:
    st.markdown(f"**Current Status:** `{case['status']}`")
    st.markdown(f"**Outcome:** `{case['final_outcome']}`")

st.markdown("---")

# Section: Explainable AI SHAP Attribution
st.subheader("🧠 Model Explainability & SHAP Risk Drivers")
st.caption("SHAP attributions reveal the specific PCA features pushing the fraud score higher or lower for this case.")

with st.spinner("Generating SHAP feature attributions..."):
    try:
        explanation = shap_explainer.explain_transaction(case["features"], top_n=6)
        
        # Narrative box
        st.markdown(
            f"""
            <div class="system-notice">
                <strong>💡 Investigator Summary:</strong> {explanation['narrative_explanation']}
            </div>
            """,
            unsafe_allow_html=True
        )

        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            st.markdown("##### 🔺 Top Risk-Increasing Drivers")
            if explanation["top_risk_increasing"]:
                df_inc = pd.DataFrame(explanation["top_risk_increasing"])
                fig_inc = px.bar(
                    df_inc,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=[DANGER_RED],
                    labels={"shap_value": "SHAP Impact (+Risk)", "feature": "Feature"}
                )
                fig_inc.update_layout(get_plotly_layout_defaults())
                st.plotly_chart(fig_inc, use_container_width=True)
            else:
                st.write("No strong risk-increasing drivers detected.")

        with col_shap2:
            st.markdown("##### 🔻 Top Risk-Decreasing Mitigators")
            if explanation["top_risk_decreasing"]:
                df_dec = pd.DataFrame(explanation["top_risk_decreasing"])
                fig_dec = px.bar(
                    df_dec,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=[SUCCESS_GREEN],
                    labels={"shap_value": "SHAP Impact (-Risk)", "feature": "Feature"}
                )
                fig_dec.update_layout(get_plotly_layout_defaults())
                st.plotly_chart(fig_dec, use_container_width=True)
            else:
                st.write("No strong risk-mitigating features detected.")

    except Exception as e:
        st.warning(f"Could not compute SHAP explanation: {e}")

st.markdown("---")

# Section: Investigation Actions & Notes
st.subheader("📝 Investigator Action & Disposition")

status_list = ["New", "Under Review", "Confirmed Fraud", "Legitimate", "Escalated"]
outcome_list = ["Pending", "Chargeback Initiated", "Account Suspended", "Cleared - False Positive", "Customer Verified"]

current_status_idx = status_list.index(case["status"]) if case["status"] in status_list else 0
current_outcome_idx = outcome_list.index(case["final_outcome"]) if case["final_outcome"] in outcome_list else 0

with st.form("investigation_form"):
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        new_status = st.selectbox("Update Case Status:", status_list, index=current_status_idx)
    with act_col2:
        new_outcome = st.selectbox("Assign Final Outcome:", outcome_list, index=current_outcome_idx)

    notes = st.text_area(
        "Investigator Case Notes & Findings:",
        value=case["notes"],
        placeholder="Enter forensic findings, customer contact logs, or cardholder verification history...",
        height=140
    )

    submitted = st.form_submit_button("💾 Save Case Investigation", use_container_width=True)

    if submitted:
        with get_db_session() as session:
            update_investigation_record(
                session,
                transaction_id=case["id"],
                status=new_status,
                notes=notes,
                outcome=new_outcome
            )
        st.success(f"Investigation record for {case['reference']} updated and saved to SQL database!")
        st.rerun()
