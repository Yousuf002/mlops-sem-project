@echo off
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5000 --cors-allowed-origins "*" --allowed-hosts "*"
pause