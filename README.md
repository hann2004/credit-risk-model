
# Credit Risk Model: Hybrid Explainability & Dashboard

![Dashboard Demo](reports/figures/dashboard.gif)

## Project Overview
This repository implements a robust credit risk modeling pipeline with reduced target leakage, hybrid explainability, and a user-friendly Streamlit dashboard. The project combines temporal data splitting, API fallback, and SHAP-based explainability to deliver both technical and accessible risk insights.

## Features
- **Temporal Data Splitting**: Prevents target leakage by using time-aware train/test splits.
- **Hybrid Explainability**: SHAP global, local, and pie chart visuals for both technical and non-technical users.
- **Streamlit Dashboard**: Interactive dashboard with API fallback, batch scoring, and explainability section.
- **API Integration**: FastAPI backend for model inference and batch processing.
- **MLflow Tracking**: Model artifacts and experiment tracking.

## Installation & Setup
1. Clone the repository:
	 ```bash
	 git clone <your-repo-link>
	 cd credit-risk-model
	 ```
2. Create a virtual environment and install dependencies:
	 ```bash
	 python3 -m venv .venv
	 source .venv/bin/activate
	 pip install -r requirements.txt
	 ```
3. Run MLflow tracking server (optional):
	 ```bash
	 mlflow ui
	 ```

## Usage Instructions
- **Data Processing**:
	```bash
	python -m src.data_processing
	```
- **Model Training**:
	```bash
	python -m src.train
	```
- **Run API**:
	```bash
	python -m uvicorn src.api.main:app --port 8000
	```
- **Launch Dashboard**:
	```bash
	streamlit run app/streamlit_app.py
	```

## Demo
See the dashboard in action:

![Dashboard Demo](reports/figures/dashboard.gif)

## File Structure
```
credit-risk-model/
├── src/
│   ├── data/temporal.py
│   ├── data_processing.py
│   ├── explainability.py
│   ├── predict.py
│   ├── api/
│   │   ├── main.py
│   │   ├── pydantic_models.py
│   ├── train.py
├── app/
│   └── streamlit_app.py
├── reports/
│   ├── figures/
│   │   └── dashboard.gif
│   └── gap_analysis_and_improvement_plan.md
├── requirements.txt
├── README.md
```

## Contributions
- Implemented temporal split to reduce target leakage.
- Built Streamlit dashboard with API fallback and batch scoring.
- Added SHAP explainability (global, local, pie chart) for technical and non-technical users.
- Improved input validation and usability.
- Committed MLflow artifacts and refreshed documentation.

## License
MIT License.

