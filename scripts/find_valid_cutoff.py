
import pandas as pd
from datetime import timedelta
from src.data.rfm import compute_rfm
from sklearn.preprocessing import StandardScaler


raw_df = pd.read_csv('data/raw/data.csv')
dates = pd.to_datetime(raw_df['TransactionStartTime'])

min_date = dates.min()
max_date = dates.max()

results = []

for cutoff in pd.date_range(min_date + timedelta(days=30), max_date - timedelta(days=14), freq='7D'):
    for outcome_days in [45, 60, 90]:
        test_raw = raw_df[(dates >= cutoff) & (dates < cutoff + timedelta(days=outcome_days))]
        if len(test_raw) == 0:
            continue
        rfm = compute_rfm(test_raw, snapshot_date=cutoff + timedelta(days=outcome_days))
        scaler = StandardScaler()
        try:
            rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
        except Exception:
            continue
        # Proxy target logic: composite risk score
        rfm["risk_score"] = (
            -rfm_scaled[:, 0]  # recency: lower is riskier
            + rfm_scaled[:, 1]  # frequency: higher is riskier
            - rfm_scaled[:, 2]  # monetary: lower is riskier
        )
        # Auto-adjust threshold to ensure at least 2 high risk
        min_high_risk = 2
        quantile = 0.85
        while quantile > 0.0:
            threshold = rfm["risk_score"].quantile(quantile)
            rfm["is_high_risk"] = (rfm["risk_score"] >= threshold).astype(int)
            if rfm["is_high_risk"].sum() >= min_high_risk:
                break
            quantile -= 0.05
        else:
            # fallback: label the single highest as high risk
            max_idx = rfm["risk_score"].idxmax()
            rfm["is_high_risk"] = 0
            rfm.loc[max_idx, "is_high_risk"] = 1
        counts = rfm["is_high_risk"].value_counts().to_dict()
        # Only keep if both classes have at least 2 samples
        if counts.get(0, 0) >= 2 and counts.get(1, 0) >= 2:
            results.append((cutoff, outcome_days, counts.get(0, 0), counts.get(1, 0)))

if results:
    print('Cutoff date | Outcome window | #Low risk | #High risk')
    for cutoff, window, n0, n1 in results:
        print(f'{cutoff.date()} | {window} days | {n0} | {n1}')
    best = max(results, key=lambda x: x[3])  # Most high risk
    print(f'\nRecommended: cutoff={best[0].date()}, outcome_window={best[1]} days, high_risk={best[3]}, low_risk={best[2]}')
else:
    print('No valid cutoff/outcome window found with both classes present.')
