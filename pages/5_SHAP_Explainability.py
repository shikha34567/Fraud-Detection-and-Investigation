"""
Page 5: Explainable AI (SHAP) Dashboard.
Global feature importance rankings, summary distributions, and localized transaction waterfall explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.database.connection import get_db_session
from src.database.crud import get_transactions_list
from src.explainability.shap_explainer import shap_explainer
from src.data.load_data import load_dataset
from src.utils.ui_components import (
    apply_custom_theme,
    get_risk_badge_html,
    get_plotly_layout_defaults,
    ACCENT_BLUE,
    DANGER_RED,
    SUCCESS_GREEN
)

st.set_page_config(page_title="SHAP Explainability | FinGuard", page_icon="🧠", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">🧠 Explainable AI (SHAP) Intelligence</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Deconstructing black-box machine learning decisions with game-theoretic Shapley attributions
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="system-notice">
        <strong>⚖️ Interpretability & Causality Disclaimer:</strong> SHAP (SHapley Additive exPlanations) values quantify
        the relative mathematical attribution of each input feature toward the model's posterior fraud probability.
        They describe <em>how the model arrived at its decision</em>, not definitive proof of external physical or fraudulent causation.
    </div>
    """,
    unsafe_allow_html=True
)

tab_global, tab_local = st.tabs(["🌐 Global Feature Importance", "🔍 Individual Transaction Attribution"])

with tab_global:
    st.subheader("Global Feature Attribution (Mean |SHAP|)")
    st.caption("Aggregated importance of features across background transactions.")

    @st.cache_data(show_spinner=False)
    def get_cached_global_importance():
        df_sample, _ = load_dataset(use_demo_if_missing=True)
        return shap_explainer.compute_global_importance(df_sample, max_rows=100)

    try:
        with st.spinner("Computing global SHAP feature importance..."):
            df_imp = get_cached_global_importance()

        fig_global = px.bar(
            df_imp.head(15),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Viridis",
            labels={"mean_abs_shap": "Mean |SHAP Value| (Impact on Model Output)", "feature": "Feature"},
            title="Top 15 Most Influential Features in Fraud Classification"
        )
        fig_global.update_layout(get_plotly_layout_defaults())
        fig_global.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_global, use_container_width=True)

        st.caption("Key Insights: In the Credit Card Fraud dataset, features such as V14, V17, V12, V4, and Amount consistently rank as decisive signals separating legitimate behavior from fraudulent patterns.")
    except Exception as e:
        st.error(f"Error computing global SHAP: {e}")

with tab_local:
    st.subheader("Individual Transaction Forensic Breakdown")

    with get_db_session() as session:
        stored_txns = get_transactions_list(session, limit=150)

    if not stored_txns:
        st.info("No recorded transactions found. Run the simulator in **6_Transaction_Simulator** first.")
    else:
        case_options = [
            f"{t['reference']} | Score: {t['risk_score']:.1f} | {t['risk_category']}"
            for t in stored_txns
        ]
        chosen_option = st.selectbox("Select Transaction to Inspect:", case_options, index=0)
        chosen_ref = chosen_option.split(" | ")[0]
        chosen_case = next(t for t in stored_txns if t["reference"] == chosen_ref)

        st.markdown(f"**Transaction:** `{chosen_case['reference']}` | **Amount:** `${chosen_case['amount']:.2f}` | **Fraud Prob:** `{chosen_case['fraud_probability']:.4f}` | **Risk Score:** `{chosen_case['risk_score']:.1f}`")

        with st.spinner("Calculating Shapley attributions for this transaction..."):
            local_exp = shap_explainer.explain_transaction(chosen_case["features"], top_n=8)

        st.markdown(
            f"""
            <div class="system-notice">
                <strong>Narrative Summary:</strong> {local_exp['narrative_explanation']}
            </div>
            """,
            unsafe_allow_html=True
        )

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("##### 🔺 Positive Contributors (Pushed Risk UP)")
            if local_exp["top_risk_increasing"]:
                df_p = pd.DataFrame(local_exp["top_risk_increasing"])
                fig_p = px.bar(
                    df_p,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=[DANGER_RED],
                    labels={"shap_value": "+SHAP Value (Elevates Risk)", "feature": "Feature"}
                )
                fig_p.update_layout(get_plotly_layout_defaults())
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.write("No positive risk drivers for this transaction.")

        with col_neg:
            st.markdown("##### 🔻 Negative Contributors (Pushed Risk DOWN)")
            if local_exp["top_risk_decreasing"]:
                df_n = pd.DataFrame(local_exp["top_risk_decreasing"])
                fig_n = px.bar(
                    df_n,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=[SUCCESS_GREEN],
                    labels={"shap_value": "-SHAP Value (Lowers Risk)", "feature": "Feature"}
                )
                fig_n.update_layout(get_plotly_layout_defaults())
                st.plotly_chart(fig_n, use_container_width=True)
            else:
                st.write("No mitigating negative drivers.")
