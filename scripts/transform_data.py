import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import PROCESSED_DATA_PATH, REPORT_PATH
except ImportError:
    # Fallback values
    PROCESSED_DATA_PATH = '/opt/airflow/data/processed'
    REPORT_PATH = '/opt/airflow/data/reports'

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    print("Warning: ydata-profiling not available, skipping profiling report")
    PROFILING_AVAILABLE = False

def create_features(df):
    """Create time-series features for stock prediction"""
    
    df = df.copy()
    
    # 1. Lag features (previous values)
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # 2. Rolling statistics
    for window in [5, 10, 20]:
        df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
    
    # 3. Price changes and returns
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
    df['returns'] = df['close'].pct_change()
    
    # 4. Volatility (high-low range)
    df['volatility'] = df['high'] - df['low']
    df['volatility_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # 5. Time-based features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # 6. Technical indicators
    # Simple Moving Average (SMA)
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Average (EMA)
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 7. Target variable (predict next close price)
    df['target'] = df['close'].shift(-1)
    
    return df

def transform_data(raw_filepath):
    """Transform raw data and create features"""
    
    print(f"\n{'='*60}")
    print(f"Data Transformation")
    print(f"{'='*60}")
    print(f"Input: {raw_filepath}\n")
    
    # Load raw data
    df = pd.read_csv(raw_filepath, index_col=0, parse_dates=True)
    print(f"✓ Loaded raw data: {df.shape}")
    
    # Create features
    print(f"✓ Creating features...")
    df_transformed = create_features(df)
    
    # Remove rows with NaN values (due to lag and rolling features)
    rows_before = len(df_transformed)
    df_transformed = df_transformed.dropna()
    rows_after = len(df_transformed)
    print(f"✓ Removed {rows_before - rows_after} rows with NaN values")
    
    print(f"✓ Final transformed data shape: {df_transformed.shape}")
    print(f"✓ Features created: {len(df_transformed.columns)} columns")
    
    # Save processed data
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filepath = f"{PROCESSED_DATA_PATH}/processed_data_{timestamp}.csv"
    df_transformed.to_csv(processed_filepath)
    print(f"✅ Processed data saved: {processed_filepath}")
    
    print(f"{'='*60}\n")
    
    # Return only the processed filepath (profiling done separately in DAG)
    return processed_filepath

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transform_data.py <raw_filepath>")
        sys.exit(1)
    
    transform_data(sys.argv[1])