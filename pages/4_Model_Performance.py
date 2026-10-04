"""
Page 4: Model Performance & Comparative Evaluation.
Displays class-imbalance-appropriate evaluation metrics (PR-AUC, F1, Recall, Precision),
ROC & PR curves, confusion matrices, threshold trade-offs, and business cost simulations.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.settings import PATHS
from src.models.evaluate import compute_threshold_analysis, simulate_business_costs
from src.utils.ui_components import (
    apply_custom_theme,
    render_metric_card,
    get_plotly_layout_defaults,
    ACCENT_BLUE,
    SUCCESS_GREEN,
    DANGER_RED,
    WARNING_AMBER
)

st.set_page_config(page_title="Model Performance | FinGuard", page_icon="📈", layout="wide")
apply_custom_theme()

st.markdown(
    """
    <div class="header-banner">
        <h2 style="margin: 0; font-weight: 700; color: #FFFFFF;">📈 Machine Learning Performance & Benchmarks</h2>
        <p style="margin: 4px 0 0 0; color: #94A3B8;">
            Comparative model evaluation under extreme class imbalance, threshold tuning, and business impact simulation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load evaluation report
report_path = PATHS.REPORTS_DIR / "model_evaluation_report.json"
if not report_path.exists():
    st.warning("Model evaluation report not found. Run `python -m src.models.train` in terminal to train and evaluate.")
    st.stop()

with open(report_path, "r") as f:
    report = json.load(f)

comparison = report.get("comparison", {})
best_model_name = report.get("best_model", "XGBoost")
is_demo = report.get("is_demo", False)

if is_demo:
    st.markdown(
        """
        <div class="system-notice">
            <strong>⚠️ Demonstration Data Notice:</strong> The metrics below were evaluated on a synthetic benchmark dataset
            generated for UI testing. When you place the real Kaggle credit card dataset in <code>data/raw/creditcard.csv</code>
            and re-train, actual production benchmarks will populate here.
        </div>
        """,
        unsafe_allow_html=True
    )

# Why Accuracy Fails Educational Alert
st.markdown(
    """
    <div class="system-notice">
        <strong>💡 Why Accuracy is Misleading in Financial Fraud:</strong> In the ULB dataset, fraudulent transactions
        account for only ~0.17% of total volume (over 99.8% legitimate). A trivial 'dummy' classifier that predicts
        every transaction is legitimate would achieve an impressive <strong>99.83% accuracy</strong>, yet detect
        <strong>0% of fraudulent attacks</strong>. Therefore, our evaluation prioritizes <strong>PR-AUC (Precision-Recall Area)</strong>,
        <strong>Fraud Recall (Sensitivity)</strong>, and <strong>F1-score</strong>.
    </div>
    """,
    unsafe_allow_html=True
)

# Model Comparison Table
st.subheader("🏆 Model Benchmark Comparison")
comparison_rows = []
for model_name, res in comparison.items():
    m = res["metrics"]
    comparison_rows.append({
        "Model": model_name,
        "PR-AUC (Primary)": m["pr_auc"],
        "Recall (Fraud)": m["recall"],
        "Precision (Fraud)": m["precision"],
        "F1-Score": m["f1_score"],
        "ROC-AUC": m["roc_auc"],
        "False Positive Rate": m["false_positive_rate"]
    })

df_comp = pd.DataFrame(comparison_rows).sort_values(by="PR-AUC (Primary)", ascending=False)
st.dataframe(
    df_comp,
    column_config={
        "PR-AUC (Primary)": st.column_config.NumberColumn(format="%.4f"),
        "Recall (Fraud)": st.column_config.NumberColumn(format="%.4f"),
        "Precision (Fraud)": st.column_config.NumberColumn(format="%.4f"),
        "F1-Score": st.column_config.NumberColumn(format="%.4f"),
        "ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
        "False Positive Rate": st.column_config.NumberColumn(format="%.4f")
    },
    hide_index=True,
    use_container_width=True
)

st.markdown("---")

# Selected Model Deep Dive
selected_model = st.selectbox("Inspect Detailed Curves & Matrix for:", list(comparison.keys()), index=0)
model_data = comparison[selected_model]
cm = model_data["confusion_matrix"]
m = model_data["metrics"]

c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
with c_kpi1:
    render_metric_card("Precision-Recall AUC", f"{m['pr_auc']:.4f}", "Area under PR curve")
with c_kpi2:
    render_metric_card("Fraud Recall", f"{m['recall']:.2%}", "Fraud cases successfully detected")
with c_kpi3:
    render_metric_card("Fraud Precision", f"{m['precision']:.2%}", "Flagged cases that are true fraud")
with c_kpi4:
    render_metric_card("F1-Score", f"{m['f1_score']:.4f}", "Harmonic mean of precision & recall")

st.markdown("<br>", unsafe_allow_html=True)

# Confusion Matrix and Curves
chart_c1, chart_c2 = st.columns(2)
layout_def = get_plotly_layout_defaults()

with chart_c1:
    st.subheader("Confusion Matrix")
    cm_matrix = [
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]]
    ]
    fig_cm = px.imshow(
        cm_matrix,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=["Legitimate (0)", "Fraud (1)"],
        y=["Legitimate (0)", "Fraud (1)"],
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(layout_def)
    st.plotly_chart(fig_cm, use_container_width=True)

with chart_c2:
    st.subheader("Precision-Recall Curve")
    pr_data = model_data.get("pr_curve", {})
    if pr_data and "recall" in pr_data and "precision" in pr_data:
        fig_pr = px.line(
            x=pr_data["recall"],
            y=pr_data["precision"],
            labels={"x": "Recall (Sensitivity)", "y": "Precision"},
            title=f"PR Curve (PR-AUC = {m['pr_auc']:.4f})"
        )
        fig_pr.update_layout(layout_def)
        st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("PR curve coordinates not available.")

st.markdown("---")

# Section: Business Cost Simulation & Decision Threshold Analysis
st.subheader("💼 Decision Threshold & Business Cost Simulation")
st.caption("Adjust operational parameters to find the optimal trade-off between customer friction (False Positives) and unmitigated fraud loss (False Negatives).")

b_col1, b_col2, b_col3 = st.columns(3)
with b_col1:
    cost_fp = st.slider("Cost per False Positive ($)", min_value=1.0, max_value=100.0, value=15.0, step=1.0,
                        help="Cost of analyst review, SMS verification, or customer friction.")
with b_col2:
    cost_fn = st.slider("Cost per False Negative ($)", min_value=50.0, max_value=2000.0, value=300.0, step=25.0,
                        help="Average unrecoverable direct loss from a missed fraudulent transaction.")
with b_col3:
    st.caption("Business Trade-Off Insight:")
    st.write(f"Ratio: **1 missed fraud = {cost_fn / cost_fp:.1f} false positives** in dollar equivalence.")

# Calculate hypothetical threshold analysis
thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
cost_rows = []
for t in thresholds:
    # Approximate based on model performance
    tp = cm["tp"]
    fp = int(cm["fp"] * (1.0 - t + 0.1) / 0.6) if t < 0.5 else int(cm["fp"] * (1.0 - t) / 0.5)
    fn = int(cm["fn"] * (t + 0.1) / 0.6) if t > 0.5 else int(cm["fn"] * t / 0.5)
    fp_cost = fp * cost_fp
    fn_cost = fn * cost_fn
    total = fp_cost + fn_cost

    cost_rows.append({
        "Threshold": t,
        "Estimated False Positives": fp,
        "Estimated False Negatives": fn,
        "Investigation Cost ($)": round(fp_cost, 2),
        "Fraud Loss ($)": round(fn_cost, 2),
        "Total Financial Impact ($)": round(total, 2)
    })

df_costs = pd.DataFrame(cost_rows)

fig_cost = px.line(
    df_costs,
    x="Threshold",
    y=["Investigation Cost ($)", "Fraud Loss ($)", "Total Financial Impact ($)"],
    markers=True,
    labels={"value": "Total Cost ($)", "Threshold": "Decision Threshold"}
)
fig_cost.update_layout(layout_def)
st.plotly_chart(fig_cost, use_container_width=True)

st.dataframe(df_costs, hide_index=True, use_container_width=True)
