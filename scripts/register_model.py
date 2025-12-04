# scripts/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

# Your MLflow server
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("stock-price-prediction")
if not experiment:
    print("Experiment 'stock-price-prediction' not found!")
    exit(1)

experiment_id = experiment.experiment_id
print(f"Found experiment ID: {experiment_id}")

# Search for the best run (lowest test_rmse)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.test_rmse ASC"],
    max_results=1
)

if not runs:
    print("No runs found in experiment!")
    exit(1)

best_run = runs[0]
run_id = best_run.info.run_id
test_rmse = best_run.data.metrics.get("test_rmse", "N/A")

print(f"Best run ID: {run_id}")
print(f"Test RMSE: {test_rmse:.4f}")

# Register model
model_name = "stock-price-predictor"

# Create registered model if not exists
try:
    client.create_registered_model(model_name)
    print(f"Created new registered model: {model_name}")
except:
    print(f"Model {model_name} already exists, adding new version...")

# Create model version from the best run
source = f"runs:/{run_id}/model"
model_version = client.create_model_version(
    name=model_name,
    source=source,
    run_id=run_id,
    tags={"version": "v1.0.0", "phase": "final"}
)

print(f"Created model version {model_version.version}")

# Transition to Production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model '{model_name}' version {model_version.version} → Production")
print("Go to http://localhost:5001 → Model Registry → You will see it!")