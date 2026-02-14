# Credit Risk Model - Gap Analysis and Improvement Plan

Date: 2026-02-13
Branch: Improvement

## Project Selection Justification
This project targets credit risk assessment, a core finance domain where reliability and transparency are critical. The repository already includes a working ML pipeline, MLflow tracking, and an API scaffold, which provides a solid base for production-grade improvements. It also has clear room for engineering upgrades such as modularity, testing, explainability, and stakeholder-facing delivery. These gaps make it an ideal capstone for demonstrating rigorous, finance-ready machine learning.

## Gap Analysis Checklist

| Category | Question | Status (Yes/No/Partial) | Evidence / Notes |
| --- | --- | --- | --- |
| Code Quality | Is the code modular and well-organized? | Partial | Core logic in src/, but mixed responsibilities and empty predict.py. |
| Code Quality | Are there type hints on functions? | Partial | Some hints exist, not consistent across modules. |
| Code Quality | Is there a clear project structure? | Partial | Basic structure exists; lacks dedicated config/util modules. |
| Testing | Are there unit tests for core functions? | Partial | 3 tests in tests/test_data_processing.py. |
| Testing | Do tests run automatically on push? | Partial | CI runs only on main branch. |
| Documentation | Is the README comprehensive? | No | Missing business impact, quick start, demo, structure, and results. |
| Documentation | Are there docstrings on functions? | Partial | Some docstrings exist, not consistent. |
| Reproducibility | Can someone else run this project? | Partial | Docker and requirements exist, but README lacks clear steps. |
| Reproducibility | Are dependencies in requirements.txt? | Yes | requirements.txt present. |
| Visualization | Is there an interactive way to explore results? | No | No dashboard present. |
| Business Impact | Is the problem clearly articulated? | Partial | README has business context but not an impact summary. |
| Business Impact | Are success metrics defined? | No | No explicit success metrics in README. |

## Improvement Plan (Prioritized)

1) Engineering Refactor (2 days)
- Add consistent type hints, dataclasses for config, and named constants.
- Split data processing, feature engineering, and labeling into dedicated modules.
- Create a minimal prediction module (predict.py) for reusable inference logic.

2) Testing + CI Hardening (1.5 days)
- Expand to at least 5 unit tests and 1 integration test.
- Update CI to run on all branches and on PRs.
- Add CI badge to README.

3) Explainability (1 day)
- Add global feature importance and SHAP explanations.
- Save plots for use in README and dashboard.

4) Streamlit Dashboard (1.5 days)
- Build a simple interface for prediction, probability display, and explainability.
- Provide a demo link or screenshots.

5) Documentation + Storytelling (1 day)
- Rewrite README with business problem, solution, results, quick start, and demo.
- Add project structure tree and reproducibility steps.
- Prepare a short technical report or blog post.

## Estimated Timeline (7 Days)

- Day 1: Gap analysis, plan, repo cleanup checklist
- Day 2: Refactor and modularize core pipeline
- Day 3: Tests and reliability fixes
- Day 4: CI updates and badge
- Day 5: Dashboard MVP
- Day 6: Explainability + documentation
- Day 7: Final polish and presentation materials
