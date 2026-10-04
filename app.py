"""
Real-Time Financial Fraud Detection & Investigation System.
Main Streamlit Application Entrypoint.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import json

from config.settings import PATHS, RISK_THRESHOLDS, MODEL_CONFIG
from src.database.connection import init_db, get_db_session
from src.database.crud import get_dashboard_summary_metrics
from src.models.predict import predictor
from src.data.load_data import DataLoader
from src.utils.ui_components import (
    apply_custom_theme,
    render_metric_card,
    get_risk_badge_html,
    ACCENT_BLUE,
    TEXT_MUTED
)

# Page Configuration
st.set_page_config(
    page_title="FinGuard | Financial Fraud Detection & Investigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Modern CSS Theme
apply_custom_theme()

# Ensure Database is Initialized
init_db()


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ **FinGuard AI Engine**")
        st.caption("Real-Time Fraud Detection & Investigation Platform")
        st.markdown("---")

        # System Status Indicators
        st.subheader("System Status")
        
        # Check Dataset
        data_loader = DataLoader()
        data_available = data_loader.is_dataset_available()
        if data_available:
            st.success("🟢 Raw Dataset: ULB Real Data")
        else:
            st.warning("🟡 Raw Dataset: Demo Fallback")

        # Check Model Artifacts
        models_available = predictor.check_artifacts_available()
        if models_available:
            # Ensure in memory
            if not predictor.is_loaded:
                predictor.load_artifacts()
            st.success(f"🟢 ML Pipeline: Ready")
            
            # Load metadata if exists
            meta_path = PATHS.MODELS_DIR / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    st.caption(f"Active Model: **{meta.get('best_model_name', 'XGBoost')}**")
        else:
            st.error("🔴 Model Artifacts: Missing")
            st.caption("Run in terminal:\n`python -m src.models.train`")

        st.markdown("---")
        st.subheader("Configured Risk Thresholds")
        st.caption(f"• **Low Risk:** 0.0 - {RISK_THRESHOLDS.LOW_UPPER:.0f}")
        st.caption(f"• **Medium Risk:** {RISK_THRESHOLDS.LOW_UPPER:.0f} - {RISK_THRESHOLDS.MEDIUM_UPPER:.0f}")
        st.caption(f"• **High Risk:** {RISK_THRESHOLDS.MEDIUM_UPPER:.0f} - {RISK_THRESHOLDS.HIGH_UPPER:.0f}")
        st.caption(f"• **Critical Risk:** {RISK_THRESHOLDS.HIGH_UPPER:.0f} - 100")

        st.markdown("---")
        st.caption("© 2026 FinGuard Analytics | Built with XGBoost, SHAP, SQLite & Streamlit")

    # Header Section
    st.markdown(
        """
        <div class="header-banner">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; color: #FFFFFF;">
                🛡️ Real-Time Financial Fraud Detection & Investigation
            </h1>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 1.05rem;">
                Machine Learning Classification, Isolation Forest Outliers, SHAP Explainability & Real-Time Case Management
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Important Investigation Disclaimer Alert
    st.markdown(
        """
        <div class="system-notice">
            <strong>⚠️ Investigation Support Notice:</strong> This platform is designed for fraud analyst decision support,
            anomaly surveillance, and risk scoring. Machine learning predictions represent statistical risk probabilities
            and do not constitute definitive proof or legal determinations of fraud.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Metrics Overview
    with get_db_session() as session:
        metrics = get_dashboard_summary_metrics(session)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            "Transactions Analyzed",
            f"{metrics['total_transactions']:,}",
            "Persisted in audit database"
        )
    with col2:
        render_metric_card(
            "Flagged Transactions",
            f"{metrics['flagged_transactions']:,}",
            "High & Critical Risk alerts",
            delta_color="inverse"
        )
    with col3:
        render_metric_card(
            "Model Predictions",
            f"{metrics['predicted_fraud_count']:,}",
            "Predicted fraud (p >= 0.50)"
        )
    with col4:
        rate = (
            (metrics['predicted_fraud_count'] / metrics['total_transactions'] * 100)
            if metrics['total_transactions'] > 0 else 0.0
        )
        render_metric_card(
            "Fraud Ratio (Stream)",
            f"{rate:.2f}%",
            "Observed in processed stream"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Navigation Cards
    st.subheader("Platform Modules & Investigation Workflow")
    nav_col1, nav_col2, nav_col3 = st.columns(3)

    with nav_col1:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 180px;">
                <h4 style="color: {ACCENT_BLUE}; margin-top: 0;">📊 Executive Dashboard</h4>
                <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">
                    Monitor transaction volume, risk tier distributions, fraud probability shifts, and recent alerts.
                </p>
                <p>👉 <em>Go to <strong>1_Dashboard</strong></em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav_col2:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 180px;">
                <h4 style="color: {ACCENT_BLUE}; margin-top: 0;">🔎 Transaction Monitoring</h4>
                <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">
                    Search, filter, and inspect incoming transactions with dynamic risk badges and CSV exports.
                </p>
                <p>👉 <em>Go to <strong>2_Transaction_Monitoring</strong></em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav_col3:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 180px;">
                <h4 style="color: {ACCENT_BLUE}; margin-top: 0;">🕵️ Fraud Investigation</h4>
                <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">
                    Examine suspicious cases, view SHAP attribution drivers, write notes, and log final outcomes.
                </p>
                <p>👉 <em>Go to <strong>3_Fraud_Investigation</strong></em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    nav_col4, nav_col5, nav_col6 = st.columns(3)
    st.markdown("<br>", unsafe_allow_html=True)

    with nav_col4:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 180px;">
                <h4 style="color: {ACCENT_BLUE}; margin-top: 0;">📈 Model Performance</h4>
                <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">
                    Compare XGBoost, Random Forest, & Logistic Regression. Explore PR-AUC, ROC, and Business Cost Simulators.
                </p>
                <p>👉 <em>Go to <strong>4_Model_Performance</strong></em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav_col5:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 180px;">
                <h4 style="color: {ACCENT_BLUE}; margin-top: 0;">🧠 SHAP Explainability</h4>
                <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">
                    Understand model reasoning. Global feature rankings, localized waterfall explanations, and directionality.
                </p>
                <p>👉 <em>Go to <strong>5_SHAP_Explainability</strong></em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav_col6:
        st.markdown(
            f"""
            <div class="metric-card" style="min-height: 180px;">
                <h4 style="color: {ACCENT_BLUE}; margin-top: 0;">⚡ Transaction Simulator</h4>
                <p style="color: {TEXT_MUTED}; font-size: 0.9rem;">
                    Stream simulated financial transactions in real-time, trigger live scoring, and watch alert generation.
                </p>
                <p>👉 <em>Go to <strong>6_Transaction_Simulator</strong></em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Quick Transaction Scorer Sandbox
    st.subheader("⚡ Quick Transaction Risk Analyzer")
    st.caption("Test individual transaction parameters directly through the active machine learning pipeline.")

    preset = st.selectbox(
        "Load Preset Example:",
        ["Standard Normal Purchase", "Suspicious Outlier Transaction", "High-Value Fraud Signature"]
    )

    # Defaults
    if preset == "High-Value Fraud Signature":
        default_amt = 1850.00
        default_time = 45000.0
        v14_val = -8.50
        v17_val = -6.20
        v12_val = -7.10
        v4_val = 5.40
    elif preset == "Suspicious Outlier Transaction":
        default_amt = 740.00
        default_time = 12000.0
        v14_val = -3.80
        v17_val = -2.90
        v12_val = -2.50
        v4_val = 2.80
    else:
        default_amt = 48.50
        default_time = 86400.0
        v14_val = 0.12
        v17_val = -0.05
        v12_val = 0.08
        v4_val = -0.20

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        test_amt = st.number_input("Transaction Amount ($)", value=default_amt, min_value=0.0, step=10.0)
    with col_in2:
        test_time = st.number_input("Transaction Time (Seconds)", value=default_time, min_value=0.0, step=100.0)
    with col_in3:
        st.caption("PCA Anomaly Indicators:")
        st.write(f"V14: `{v14_val}` | V17: `{v17_val}` | V4: `{v4_val}`")

    if st.button("🚀 Analyze Transaction Risk", use_container_width=True):
        if not predictor.is_loaded and not predictor.load_artifacts():
            st.error("Trained models not found. Please train models first using: python -m src.models.train")
        else:
            # Construct feature vector
            feature_dict = {f"V{i}": 0.0 for i in range(1, 29)}
            feature_dict["Time"] = test_time
            feature_dict["Amount"] = test_amt
            feature_dict["V14"] = v14_val
            feature_dict["V17"] = v17_val
            feature_dict["V12"] = v12_val
            feature_dict["V4"] = v4_val

            prediction = predictor.predict_single(feature_dict)

            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            with res_c1:
                st.metric("Fraud Probability", f"{prediction['fraud_probability']:.2%}")
            with res_c2:
                st.metric("Assigned Risk Score", f"{prediction['risk_score']:.1f} / 100")
            with res_c3:
                st.markdown(f"**Risk Tier:**<br>{get_risk_badge_html(prediction['risk_category'])}", unsafe_allow_html=True)
            with res_c4:
                st.metric("Anomaly Score", f"{prediction['anomaly_score']:.3f}")


if __name__ == "__main__":
    main()
