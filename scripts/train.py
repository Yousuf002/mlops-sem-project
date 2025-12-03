import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MLFLOW_TRACKING_URI, MODEL_PATH

# ============ ADD THIS SECTION ============
# Configure S3 (MinIO) for MLflow artifact storage
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
# ==========================================

def train_model(processed_filepath):
    """Train stock price prediction model with MLflow tracking"""
    
    print(f"\n{'='*60}")
    print(f"Model Training")
    print(f"{'='*60}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Set experiment name
    experiment_name = "stock-price-prediction"
    mlflow.set_experiment(experiment_name)
    
    print(f"Loading data from: {processed_filepath}")
    df = pd.read_csv(processed_filepath, index_col=0, parse_dates=True)
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['target', 'symbol', 'extraction_time']]
    X = df[feature_columns]
    y = df['target']
    
    # Split data (don't shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"stock_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("date_range", f"{df.index.min()} to {df.index.max()}")
        mlflow.log_param("stock_symbol", df['symbol'].iloc[0] if 'symbol' in df.columns else 'N/A')
        
        # === ONLY CHANGE: create directory first + print real values ===
        os.makedirs(MODEL_PATH, exist_ok=True)   # ← THIS FIXES THE ERROR
        
        feature_mins = X_train.min().values
        feature_maxs = X_train.max().values
        
        np.save(f"{MODEL_PATH}/feature_min.npy", feature_mins)
        np.save(f"{MODEL_PATH}/feature_max.npy", feature_maxs)
        
        mlflow.log_artifact(f"{MODEL_PATH}/feature_min.npy")
        mlflow.log_artifact(f"{MODEL_PATH}/feature_max.npy")
        
        print("\n" + "="*60)
        print("FEATURE RANGES FOR OOD DETECTION (copy to main.py)")
        print("="*60)
        print("FEATURE_MIN = np.array(" + str(list(feature_mins)) + ")")
        print("FEATURE_MAX = np.array(" + str(list(feature_maxs)) + ")")
        print("="*60 + "\n")
        # ====================================================================

        # Define hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log hyperparameters
        print(f"\nModel hyperparameters:")
        for param, value in params.items():
            mlflow.log_param(param, value)
            print(f"  - {param}: {value}")
        
        # Train model
        print(f"\nTraining Random Forest model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance_path = f"{MODEL_PATH}/feature_importance.csv"
        feature_importance.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path)
        
        print(f"\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        ) 
        
        # Save model locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{MODEL_PATH}/model_{timestamp}.joblib"
        joblib.dump(model, model_filename)
        
        print(f"\n{'='*60}")
        print(f"Model Training Complete!")
        print(f"{'='*60}")
        print(f"Performance Metrics:")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:  {test_rmse:.4f}")
        print(f"  Train MAE:  {train_mae:.4f}")
        print(f"  Test MAE:   {test_mae:.4f}")
        print(f"  Train R²:   {train_r2:.4f}")
        print(f"  Test R²:    {test_r2:.4f}")
        print(f"{'='*60}")
        print(f"Model saved to: {model_filename}")
        print(f"{'='*60}\n")
        
        return model_filename, test_rmse

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <processed_filepath>")
        sys.exit(1)
    
    train_model(sys.argv[1])