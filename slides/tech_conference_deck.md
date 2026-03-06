---
marp: true
paginate: true
theme: default
size: 16:9
style: |
  section {
    background: #0D0D0D;
    color: #EAF2FF;
    font-family: Inter, Roboto, sans-serif;
    padding: 48px 64px;
    background-image:
      linear-gradient(rgba(255,255,255,0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px);
    background-size: 30px 30px;
  }

  h1, h2, h3 {
    font-family: Montserrat, Poppins, sans-serif;
    letter-spacing: 0.3px;
    color: #FFFFFF;
    margin: 0;
  }

  h1 {
    font-size: 66px;
    line-height: 1.02;
    font-weight: 800;
  }

  h2 {
    font-size: 44px;
    line-height: 1.1;
    font-weight: 760;
    margin-bottom: 16px;
  }

  h3 {
    font-size: 28px;
    margin-bottom: 8px;
  }

  p, li {
    font-size: 22px;
    line-height: 1.35;
  }

  ul {
    margin: 8px 0 0;
    padding-left: 24px;
  }

  .accent { color: #00BFFF; }
  .violet { color: #7B61FF; }
  .danger { color: #FF5C8A; }
  .muted { color: #9EB0CD; }
  .tiny { font-size: 15px; }
  .small { font-size: 18px; }
  .center { text-align: center; }

  .hero-sub {
    margin-top: 18px;
    font-size: 24px;
    color: #A9B8D0;
  }

  .footer {
    position: absolute;
    left: 64px;
    bottom: 24px;
    color: #7F8FAB;
    font-size: 15px;
  }

  .grid-two {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px;
    align-items: start;
  }

  .grid-three {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 18px;
    align-items: stretch;
  }

  .card {
    border: 1px solid rgba(0,191,255,0.42);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(18,22,32,0.62);
  }

  .card-violet {
    border: 1px solid rgba(123,97,255,0.55);
  }

  .band {
    margin-top: 16px;
    border-radius: 12px;
    padding: 12px 16px;
    background: linear-gradient(95deg, rgba(0,191,255,0.2), rgba(123,97,255,0.18));
    border: 1px solid rgba(255,255,255,0.15);
    font-weight: 650;
  }

  .divider {
    height: 1px;
    margin: 14px 0;
    background: linear-gradient(90deg, rgba(0,191,255,0.9), rgba(123,97,255,0.14));
  }

  .diagram {
    border: 1px solid rgba(123,97,255,0.58);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(12,15,24,0.7);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 18px;
    line-height: 1.35;
    white-space: pre;
  }

  .big-number {
    font-size: 112px;
    line-height: 0.9;
    font-weight: 850;
    color: #00BFFF;
  }

  .metric {
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 12px;
    padding: 10px 12px;
    background: rgba(16,20,30,0.72);
  }

  .img-card {
    border: 1px solid rgba(0,191,255,0.38);
    border-radius: 12px;
    padding: 8px;
    background: rgba(10,14,20,0.68);
  }

  .code {
    border: 1px solid rgba(123,97,255,0.6);
    border-radius: 12px;
    padding: 12px 14px;
    background: rgba(8,11,18,0.9);
    color: #DBE8FF;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 15px;
    line-height: 1.3;
    white-space: pre;
  }

  @keyframes pulseGlow {
    0% { box-shadow: 0 0 0 rgba(0,191,255,0.0); }
    50% { box-shadow: 0 0 22px rgba(0,191,255,0.3); }
    100% { box-shadow: 0 0 0 rgba(0,191,255,0.0); }
  }

  .glow {
    animation: pulseGlow 2.6s ease-in-out infinite;
  }

---

<!-- _class: center -->

# PRODUCTION-AWARE  
# CREDIT RISK SYSTEM

<div class="hero-sub">Temporal Leakage Control • MLflow Tracking • API Deployment</div>

<div class="band">Real-world transaction data • 3,194 customers • End-to-end ML system</div>

<div class="footer">Nabi • KAIM Capstone 2026</div>

---

## The Real Problem

<div class="grid-two">
<div>

# Credit scoring fails  
# thin-file customers

<div class="divider"></div>

<p class="small muted">Traditional bureau data is sparse or unavailable for many applicants.</p>

</div>
<div>

<div class="card">
- No traditional credit history
- Limited underwriting signals
- High uncertainty in lending
</div>

<div class="band">Can transaction behavior predict risk?</div>

</div>
</div>

---

## Data → Intelligence Pipeline

<div class="diagram">Raw Transactions
      ↓
Feature Engineering
      ↓
Customer Profiles
      ↓
Leakage-Controlled Risk Labels
      ↓
Production Model</div>

<div class="grid-three" style="margin-top:14px;">
<div class="metric"><span class="accent">3,194</span><br><span class="small muted">Customers</span></div>
<div class="metric"><span class="violet">59</span><br><span class="small muted">Engineered Features</span></div>
<div class="metric"><span class="accent">95k+</span><br><span class="small muted">Transactions analyzed</span></div>
</div>

<div class="small" style="margin-top:12px;">Aggregates (sum/avg/count/std) • Datetime behavior • RFM-based risk structure</div>

---

## Dashboard Experience

<div class="img-card glow">

![w:1180](assets/dashboard_frame.png)

</div>

<p class="small muted">Interactive risk scoring, thresholding, SHAP explainability, and API/local inference fallback.</p>

---

<!-- _class: center -->

## When Metrics Lie

# The Model Was Too Good

<div class="big-number">ROC-AUC ≈ 1.0</div>

<div class="hero-sub">That result looked impressive — and suspicious.</div>

---

## Root Cause: Data Leakage

<div class="grid-two">
<div class="card">
<h3>Feature Window</h3>
<p class="small muted">used for training</p>
<div class="diagram">[==== PAST+FUTURE SIGNAL ====]
                <span class="danger">OVERLAP</span></div>
</div>
<div class="card card-violet">
<h3>Target Window</h3>
<p class="small muted">used for labeling</p>
<div class="diagram"><span class="danger">OVERLAP</span>
[==== SAME PERIOD ====]</div>
</div>
</div>

<div class="band"><span class="danger">Leakage:</span> target was influenced by the same period used to construct features.</div>

---

## Temporal Split Fix (Production Design)

<div class="diagram">Past ------------------| cutoff_date |------------------ Future
   build features only                     build target only</div>

<div class="grid-three" style="margin-top:14px;">
<div class="card">Strict chronology enforced</div>
<div class="card">No future signal in features</div>
<div class="card">Realistic out-of-time behavior</div>
</div>

<div class="band">Performance remained strong even after leakage removal.</div>

---

## Model Strategy & Selection

<div class="grid-two">
<div class="card">
<h3>Logistic Regression</h3>
<p class="small">GridSearchCV tuning</p>
</div>
<div class="card card-violet">
<h3>Random Forest</h3>
<p class="small">RandomizedSearchCV tuning</p>
</div>
</div>

<div class="divider"></div>

<p class="small">Cross-validation (cv=3) • RandomOverSampler • Hyperparameter search</p>
<div class="band">Final production choice: <span class="accent">Random Forest</span></div>

---

## Model Evidence (Real Plots)

<div class="img-card glow">

![w:980](../reports/figures/best_models.png)

</div>

<div class="grid-three" style="margin-top:10px;">
<div class="card small">Model comparison clarity</div>
<div class="card small">Leakage-fix performance retained</div>
<div class="card small">Production model justified</div>
</div>

---

<!-- _class: center -->

## Final Holdout Performance (Leakage-Safe)

<div class="small muted">ROC-AUC</div>
<div class="big-number">0.9229</div>

<div class="grid-two small" style="margin-top:8px;">
<div class="metric">Accuracy <span class="accent">0.8795</span></div>
<div class="metric">Precision <span class="accent">0.5933</span></div>
<div class="metric">Recall <span class="violet">0.8476</span></div>
<div class="metric">F1-score <span class="violet">0.6980</span></div>
</div>

<div class="small muted" style="margin-top:8px;">Evaluation: stratified holdout test (20%, random_state=42)</div>

---

## Engineering & MLOps Stack

<div class="grid-three">
<div class="card">
<h3>MLflow</h3>
<p class="small">experiments<br>metrics<br>model artifacts</p>
</div>
<div class="card card-violet">
<h3>Reproducibility</h3>
<p class="small">requirements.txt<br>structured src/<br>deterministic seeds</p>
</div>
<div class="card">
<h3>Deployment</h3>
<p class="small">FastAPI /predict<br>Streamlit UI<br>JSON inference</p>
</div>
</div>

<div class="band">Designed for production behavior, observability, and reliable decision support.</div>

---

## API Snippet (Production-Ready)

<div class="grid-two">
<div class="code"># src/api/main.py
@app.post("/predict")
def predict(payload: PredictionRequest):
    probs = predict_instances(
        model,
        payload.instances,
        feature_names,
    )
    return {
      "risk_probabilities": probs
    }</div>

<div class="code">POST /predict
{
  "instances": [
    {
      "total_amount": 0.42,
      "avg_amount": -0.11,
      "txn_count": 0.63,
      "std_amount": 0.08
    }
  ]
}

→ 200 OK
{"risk_probabilities": [0.83]}</div>
</div>

<div class="small muted" style="margin-top:8px;">Health endpoint available: GET /health</div>

---

## End-to-End Runtime Architecture

<div class="diagram">User
  ↓
Streamlit Dashboard
  ↓
FastAPI Inference Layer
  ↓
Production Model (.pkl)
  ↓
Risk Probability Output</div>

<div class="grid-two" style="margin-top:12px;">
<div class="card small">Primary mode: API scoring</div>
<div class="card small">Fallback mode: local model inference</div>
</div>

---

<!-- _class: center -->

## Lessons That Matter

- High accuracy can hide leakage
- Time-aware pipelines are non-negotiable
- Production quality starts at data design
- Engineering discipline > leaderboard metrics

<div class="grid-two" style="margin-top:14px; text-align:left;">
<div class="card">
<h3>What ships today</h3>
<p class="small">Temporal leakage controls<br>API + dashboard inference<br>Reproducible training pipeline</p>
</div>
<div class="card card-violet">
<h3>What comes next</h3>
<p class="small">Out-of-time monitoring<br>Automated retraining triggers<br>Containerized cloud deployment</p>
</div>
</div>

<div class="band">Built to make risk decisions trustworthy in real operations.</div>
