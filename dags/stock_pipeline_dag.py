from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Configure paths for Airflow
sys.path.insert(0, '/opt/airflow')
sys.path.insert(0, '/opt/airflow/scripts')

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_price_prediction_pipeline',
    default_args=default_args,
    description='Real-time stock price prediction MLOps pipeline',
    schedule_interval='@daily',
    catchup=False,
)

def extract_wrapper(**context):
    """Wrapper for extract function"""
    from extract_data import extract_stock_data
    filepath = extract_stock_data()
    # Push filepath to XCom
    context['ti'].xcom_push(key='raw_filepath', value=filepath)
    return filepath

def quality_check_wrapper(**context):
    """Wrapper for quality check function"""
    from data_quality_check import perform_quality_check
    ti = context['ti']
    filepath = ti.xcom_pull(task_ids='extract_stock_data', key='raw_filepath')
    result = perform_quality_check(filepath)
    return result

def transform_wrapper(**context):
    """Wrapper for transform function"""
    from transform_data import transform_data
    ti = context['ti']
    raw_filepath = ti.xcom_pull(task_ids='extract_stock_data', key='raw_filepath')
    processed_filepath = transform_data(raw_filepath)
    ti.xcom_push(key='processed_filepath', value=processed_filepath)
    return processed_filepath

def train_wrapper(**context):
    """Wrapper for train function"""
    from train import train_model
    ti = context['ti']
    processed_filepath = ti.xcom_pull(task_ids='transform_data', key='processed_filepath')
    model_path, rmse = train_model(processed_filepath)
    return model_path

# Task 1: Extract data
extract_task = PythonOperator(
    task_id='extract_stock_data',
    python_callable=extract_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 2: Quality check
quality_check_task = PythonOperator(
    task_id='data_quality_check',
    python_callable=quality_check_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 3: Transform data
transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 4: Generate profiling report
def profiling_wrapper(**context):
    """Generate profiling report"""
    from generate_profile import generate_profile
    ti = context['ti']
    processed_filepath = ti.xcom_pull(task_ids='transform_data', key='processed_filepath')
    report_path = generate_profile(processed_filepath)
    return report_path

profiling_task = PythonOperator(
    task_id='generate_profiling_report',
    python_callable=profiling_wrapper,
    provide_context=True,
    dag=dag,
)

# Task 5: Train model
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_wrapper,
    provide_context=True,
    dag=dag,
)

# Define dependencies
extract_task >> quality_check_task >> transform_task >> profiling_task >> train_task