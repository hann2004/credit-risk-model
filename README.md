## Credit Scoring Business Understanding

In this project, we aim to build a credit risk probability model for Bati Bank in partnership with an eCommerce platform, enabling a buy-now-pay-later service. The model estimates the likelihood that a customer will default on a loan and informs decisions on approvals, terms, and amounts.

**1. Basel II and the Need for Interpretable Models**
- Basel II links model outputs directly to regulatory capital. Models must be transparent, validated, and traceable, with documented assumptions, transformations (e.g., Weight of Evidence), and governance.
- Interpretable, well-documented models improve auditability, facilitate backtesting and ongoing monitoring, and reduce model risk by making decisions explainable to regulators, auditors, and internal stakeholders.

**2. Necessity of Proxy Variables**
- The dataset lacks a direct default label, so we create a proxy using behavioral RFM signals (recency, frequency, monetary) to flag high- vs. low-risk customers.
- Risks include label bias, policy-driven changes, and data drift that can misclassify risk—impacting approvals, pricing, and capital allocation.
- Mitigate via periodic calibration, challenger analysis, stability monitoring, and backtesting against any subsequently observed defaults.

**3. Trade-offs Between Simple and Complex Models**
- Simple, interpretable models (e.g., Logistic Regression with WoE): easy to explain and document, align with regulatory expectations; may have slightly lower predictive performance.
- Complex, high-performance models (e.g., Gradient Boosting): can capture non-linear patterns and improve accuracy; harder to interpret, require explainability tooling (e.g., SHAP) and stronger validation/controls (e.g., monotonicity constraints or scorecard extraction).
- In regulated contexts, balance predictive performance, interpretability, and compliance: start with interpretable baselines, justify added complexity with measurable uplift, and maintain robust documentation, monitoring, and governance.

