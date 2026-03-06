from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "credit-risk"

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"Experiment '{EXPERIMENT_NAME}' not found.")
    exit(1)

runs = client.search_runs(experiment.experiment_id, order_by=["metrics.roc_auc DESC"])

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    for metric_name, value in run.data.metrics.items():
        print(f"{metric_name}: {value:.4f}")
    if "roc_auc" in run.data.metrics and "f1" in run.data.metrics:
        roc_auc = run.data.metrics["roc_auc"]
        f1 = run.data.metrics["f1"]
        if roc_auc >= 0.80 and f1 >= 0.75:
            print("✅ Model meets criteria!")
        else:
            print("❌ Model does NOT meet criteria.")
    print("-")
