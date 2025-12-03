#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
EOSQL

echo "MLflow database created successfully!d"
