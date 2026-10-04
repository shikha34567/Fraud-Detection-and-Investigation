"""
Page 7: Data Quality & Schema Validation Audit.
Audits missing values, duplicate records, class imbalance distributions,
and provides downloadable data quality certificates.
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px

from config.settings import PATHS
from src.data.load_data import DataLoader, load_dataset
from src.data.validate_data import DataValidator
from src.utils.ui_components import (
    apply_custom_theme,
    render_metric_card,
    get_plotly_layout_defaults,
    ACCENT_BLUE,
    SUCCESS_GREEN,
    WARNING_AMBER,
    DANGER_RED
)

st.set_page_config(page_title="Data Quality Audit | FinGuard", page_icon="📋", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">📋 Data Quality & Schema Integrity Audit</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Audit dataset dimensions, feature distributions, missingness, duplicates, and class imbalance
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

loader = DataLoader()
is_real_data = loader.is_dataset_available()

# Load and validate
raw_df, is_demo = load_dataset(use_demo_if_missing=True)
validator = DataValidator()
report = validator.generate_quality_report(raw_df, is_demo=is_demo)

# Save report
validator.save_report(report)

if is_demo:
    st.markdown(
        """
        <div class="system-notice">
            <strong>⚠️ Notice: Synthetic Demo Mode Active:</strong><br>
            The raw <code>creditcard.csv</code> file was not detected in <code>data/raw/</code>.<br>
            To use the genuine ULB Credit Card Fraud dataset (284,807 transactions):
            <ol>
                <li>Download <code>creditcard.csv</code> from <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank" style="color: #38bdf8;">Kaggle Credit Card Fraud</a>.</li>
                <li>Place the CSV file into: <code>data/raw/creditcard.csv</code>.</li>
                <li>Re-run training via terminal: <code>python -m src.models.train</code>.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.success("🟢 Genuine ULB Credit Card Dataset Detected and Validated!")

# Top KPI row
k1, k2, k3, k4 = st.columns(4)
with k1:
    render_metric_card("Total Rows", f"{report['total_rows']:,}", "Loaded records")
with k2:
    render_metric_card("Total Columns", f"{report['total_columns']}", "Features + Target")
with k3:
    render_metric_card("Missing Values", f"{report['missing_values']['total_missing']}", "Zero null tolerance")
with k4:
    render_metric_card("Duplicate Rows", f"{report['duplicate_records']:,}", "Identified in source")

st.markdown("<br>", unsafe_allow_html=True)

# Target Distribution Breakdown
st.subheader("Target Distribution & Extreme Imbalance Analysis")
target_stats = report.get("target_distribution", {})
if target_stats:
    tc1, tc2 = st.columns(2)
    layout_def = get_plotly_layout_defaults()

    with tc1:
        df_target = pd.DataFrame([
            {"Class": "Legitimate (0)", "Count": target_stats.get("legitimate_count", 0)},
            {"Class": "Fraudulent (1)", "Count": target_stats.get("fraud_count", 0)}
        ])
        fig_target = px.bar(
            df_target,
            x="Class",
            y="Count",
            color="Class",
            color_discrete_map={"Legitimate (0)": SUCCESS_GREEN, "Fraudulent (1)": DANGER_RED},
            text_auto=True,
            title=f"Class Imbalance Ratio: {target_stats.get('imbalance_ratio', 'N/A')}"
        )
        fig_target.update_layout(layout_def)
        st.plotly_chart(fig_target, use_container_width=True)

    with tc2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Class Imbalance Breakdown</h4>
                <ul>
                    <li><strong>Legitimate Transactions:</strong> {target_stats.get('legitimate_count', 0):,}</li>
                    <li><strong>Fraudulent Transactions:</strong> {target_stats.get('fraud_count', 0):,}</li>
                    <li><strong>Fraud Percentage:</strong> {target_stats.get('fraud_percentage', 0.0):.4f}%</li>
                    <li><strong>Imbalance Ratio:</strong> {target_stats.get('imbalance_ratio', 'N/A')}</li>
                </ul>
                <p style="color: #94A3B8; font-size: 0.85rem;">
                    Extreme skew requires specialized modeling strategies: cost-sensitive weighting (scale_pos_weight),
                    PR-AUC optimization, and threshold calibration.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Amount Statistics
st.markdown("---")
st.subheader("Transaction Amount Distribution")
amt_stats = report.get("amount_statistics", {})
if amt_stats:
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.metric("Min Amount", f"${amt_stats['min']:.2f}")
    with ac2:
        st.metric("Median Amount", f"${amt_stats['median']:.2f}")
    with ac3:
        st.metric("Mean Amount", f"${amt_stats['mean']:.2f}")
    with ac4:
        st.metric("Max Amount", f"${amt_stats['max']:.2f}")

# Validation Warnings & Certificate Download
st.markdown("---")
st.subheader("Validation Warnings & Data Quality Certificate")

warnings = report.get("validation_warnings", [])
if warnings:
    for w in warnings:
        st.warning(f"⚠️ {w}")
else:
    st.success("✅ All schema and integrity checks passed with zero warnings.")

# Download JSON Report
st.download_button(
    label="📥 Download Data Quality Audit Report (JSON)",
    data=json.dumps(report, indent=2),
    file_name="data_quality_report.json",
    mime="application/json",
    use_container_width=True
)
