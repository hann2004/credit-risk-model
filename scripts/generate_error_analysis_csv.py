import joblib
import pandas as pd

# Paths
DATA_PATH = "data/processed/processed_with_target.csv"
MODEL_PATH = "models/production_model.pkl"
OUTPUT_PATH = "data/processed/with_predictions_for_error_analysis.csv"

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Get expected features
feature_names = list(getattr(model, "feature_names_in_", []))


# Drop non-feature columns and align
X = df.drop(
    columns=[col for col in ["CustomerId", "is_high_risk"] if col in df.columns],
    errors="ignore",
)
X = X.reindex(columns=feature_names, fill_value=0)

# Debug prints
print("Model expects features:", feature_names)
print("CSV columns:", list(df.columns))
print("X columns after reindex:", list(X.columns))
print("X shape:", X.shape)

# Predict
y_pred = model.predict(X)

# Prepare output
df_out = pd.DataFrame()
df_out["y_true"] = df["is_high_risk"]
df_out["y_pred"] = y_pred
for col in X.columns:
    df_out[col] = X[col]

df_out.to_csv(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
