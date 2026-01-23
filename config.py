import os

# ============ DAGSHUB CONFIGURATION ============
DAGSHUB_REPO_OWNER = os.getenv('DAGSHUB_REPO_OWNER', 'Yousuf002')  # ← CHANGE THIS
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME', 'mlops-rps-stock-prediction')   # ← CHANGE THIS

# MLflow Configuration - Point to DagHub
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI', 
    f'https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow'
)

# DagHub Authentication
DAGSHUB_USER = os.getenv('DAGSHUB_USER', DAGSHUB_REPO_OWNER)
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '4a8466f05f9f3f2c2cb41a4ab9ccb5b6b1050587')  # Get token from DagHub settings
# ================================================

# MinIO Configuration (for local artifact storage - now optional with DagHub)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')

# Paths
DATA_PATH = '/opt/airflow/data'
MODEL_PATH = '/opt/airflow/models'
RAW_DATA_PATH = f'{DATA_PATH}/raw'
PROCESSED_DATA_PATH = f'{DATA_PATH}/processed'
REPORT_PATH = f'{DATA_PATH}/reports'

# API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '42MPNIZX1S8YL97E')
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'IBM')

# Data Quality Thresholds
MAX_NULL_PERCENTAGE = 1.0  # Maximum 1% null values allowed
MIN_RECORDS = 50  # Minimum 50 records required