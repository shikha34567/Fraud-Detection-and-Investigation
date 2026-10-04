# Model Card: Financial Fraud Detection & Risk Scoring System

Following the Model Cards for Model Reporting framework (Mitchell et al., 2019).

## 1. Model Details
- **Model Name:** FinGuard Fraud Classification & Anomaly Surveillance Engine
- **Model Type:** Gradient Boosted Decision Trees (XGBoost) + Isolation Forest (Outlier Detection)
- **Model Version:** `v1.0.0`
- **Release Date:** October 2026
- **Primary Frameworks:** XGBoost, Scikit-Learn, SHAP, SQLAlchemy
- **License:** MIT License

---

## 2. Intended Use
- **Primary Intended Use:** Decision-support system for bank fraud analysts, AML officers, and financial risk investigators.
- **Intended Users:** Financial institutions, payment processors, risk analysts, data scientists.
- **Key Capabilities:**
  - Real-time fraud probability estimation for card-not-present and point-of-sale transactions.
  - Assignment of normalized 0–100 risk scores and 4-tier risk categories (Low, Medium, High, Critical).
  - Identification of structural data distribution anomalies via Isolation Forest.
  - Localized feature-level attributions via TreeSHAP for compliance and investigation explainability.

---

## 3. Out-of-Scope & Prohibited Uses
- **Autonomous Legal Adjudication:** This model MUST NOT be used to unilaterally accuse individuals of criminal activity without human investigator verification.
- **Credit Scoring / Underwriting:** The model is strictly trained for transaction anomaly and fraud detection, not creditworthiness assessment.
- **Unverified Real-World Direct Blocking:** High-risk thresholds must be calibrated with human-in-the-loop review to prevent false-positive denial of essential transactions.

---

## 4. Factors, Training & Evaluation Data
- **Dataset:** ULB Machine Learning Group Credit Card Fraud Detection Dataset (284,807 transactions collected from European cardholders in September 2013).
- **Class Imbalance:** 492 fraud cases (0.172%) vs. 284,315 legitimate transactions (99.828%).
- **Features:** 30 input features including `Time`, `Amount`, and 28 principal components (`V1` to `V28`) resulting from PCA dimensionality reduction for privacy protection.
- **Data Splitting:** Chronological split (70% Train, 10% Validation, 20% Held-Out Test) based on transaction time to simulate temporal real-world deployment.

---

## 5. Evaluation Methodology & Metrics
Standard accuracy is fundamentally invalid under extreme class imbalance. Models are evaluated using:
- **PR-AUC (Average Precision):** Primary metric evaluating the precision-recall trade-off across all operating thresholds.
- **Fraud Recall (Sensitivity):** Percentage of actual fraudulent transactions intercepted.
- **Fraud Precision:** Proportion of model-flagged transactions that are confirmed fraudulent.
- **F1-Score:** Harmonic balance of precision and recall.
- **False Positive Rate (FPR):** Impact on legitimate customer transactions.

---

## 6. Explainability & Interpretability (SHAP)
- Uses `shap.TreeExplainer` to calculate local Shapley values $\phi_i$ satisfying the efficiency, symmetry, and additivity axioms:
$$\sum_{i=1}^M \phi_i = f(x) - E[f(X)]$$
- Decomposes each decision into **risk-elevating features** (+SHAP) and **mitigating features** (-SHAP).
- **Disclaimer:** SHAP explains statistical model correlation within the training distribution; it does not provide forensic or causal proof.

---

## 7. Caveats & Recommendations
1. **Concept Drift:** Fraud patterns evolve rapidly. Periodic retraining and continuous drift monitoring are mandatory.
2. **Threshold Tuning:** The decision threshold (default $0.50$, equivalent to risk score $50$) should be dynamically adjusted using the embedded Business Cost Simulator based on analyst review capacity and fraud loss appetite.
