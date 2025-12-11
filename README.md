## Credit Scoring Business Understanding

In this project, we aim to build a credit risk model that aligns with Basel II regulations, helping financial institutions estimate the likelihood of borrowers defaulting on loans and determine the necessary capital reserves.

**Basel II: Why Interpretability and Documentation Matter**
- Basel II links model outputs directly to regulatory capital. Models must be transparent, validated, and traceable, with documented assumptions, transformations (e.g., Weight of Evidence), and governance.
- Interpretable, well-documented models improve auditability, facilitate backtesting and ongoing monitoring, and reduce model risk by making decisions explainable to regulators, auditors, and internal stakeholders.

**Proxy Default Label: Necessity and Risks**
- Without a true default label, a proxy (e.g., 90+ days past due) enables supervised learning and model training.
- Risks include label bias, policy-driven changes, and data drift that can misclassify risk—impacting approvals, pricing, and capital allocation.
- Mitigate via periodic calibration, challenger analysis, stability monitoring, and backtesting against any subsequently observed defaults.

**Model Trade-offs: Simple vs. Complex**
- Simple models (e.g., Logistic Regression with WoE): high interpretability, stable governance, straightforward documentation; potentially lower predictive accuracy (AUC/KS) compared to complex methods.
- Complex models (e.g., Gradient Boosting): higher predictive power and ability to capture non-linearities; harder to explain, require explainability tooling (e.g., SHAP), more thorough validation, and stricter controls (e.g., monotonicity constraints or scorecard extraction).
- In regulated contexts, favor a balanced approach: start with interpretable baselines, justify complexity with measurable uplift, and maintain robust documentation, monitoring, and governance.

# credit-risk-model