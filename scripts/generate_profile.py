import pandas as pd
from ydata_profiling import ProfileReport
import mlflow
import os

def generate_profile(processed_filepath):
    """Generate pandas profiling report and log to MLflow"""
    
    print(f"Generating profiling report for: {processed_filepath}")
    
    # Load data
    df = pd.read_csv(processed_filepath, index_col=0, parse_dates=True)
    
    # Generate profile (minimal to save time)
    profile = ProfileReport(df, title="Stock Data Quality Report", minimal=True)
    
    # Save report
    os.makedirs("/opt/airflow/data/reports", exist_ok=True)
    report_path = "/opt/airflow/data/reports/profiling_report.html"
    profile.to_file(report_path)
    
    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Configure S3
    os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    
    with mlflow.start_run(run_name="data_profiling"):
        mlflow.log_artifact(report_path)
        mlflow.log_param("dataset_path", processed_filepath)
        mlflow.log_metric("num_rows", len(df))
        mlflow.log_metric("num_columns", len(df.columns))
    
    print(f"✓ Profiling report saved and logged to MLflow")
    return report_path