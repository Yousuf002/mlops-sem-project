import requests
import pandas as pd
from datetime import datetime
import os
import sys
import json

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALPHA_VANTAGE_API_KEY, STOCK_SYMBOL, RAW_DATA_PATH

def extract_stock_data():
    """Extract stock data from Alpha Vantage API"""
    
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    
    # API endpoint for intraday data (5-minute intervals)
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": STOCK_SYMBOL,
        "interval": "5min",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"  # Changed from "full" - this is free tier compatible
    }
    
    print(f"Fetching data for {STOCK_SYMBOL}...")
    print(f"Using free tier: Last 100 data points (5-min intervals)")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    
    data = response.json()
    
    # Debug: Print what we received
    print(f"🔍 API Response Keys: {list(data.keys())}")
    
    # Check for API errors
    if "Error Message" in data:
        raise Exception(f"API Error: {data['Error Message']}")
    
    if "Note" in data:
        print(f"\n⚠️  API Rate Limit Hit!")
        print(f"Message: {data['Note']}")
        print(f"\n💡 Alpha Vantage free tier allows only 25 requests per day.")
        print(f"💡 Waiting period: Usually resets after 1 minute")
        raise Exception(f"API Limit: {data['Note']}")
    
    if "Information" in data:
        print(f"\n⚠️  API Information: {data['Information']}")
        raise Exception(f"API Information: {data['Information']}")
    
    # Extract time series data
    time_series_key = "Time Series (5min)"
    if time_series_key not in data:
        # Print full response for debugging
        print(f"\n❌ Error: Expected key '{time_series_key}' not found")
        print(f"\n📋 Full API Response:")
        print(json.dumps(data, indent=2))
        raise Exception(f"Expected key '{time_series_key}' not found in response. Check API response above.")
    
    time_series = data[time_series_key]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Convert string values to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_index()
    
    # Add metadata
    df['symbol'] = STOCK_SYMBOL
    df['extraction_time'] = datetime.now()
    
    # Save raw data with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RAW_DATA_PATH}/stock_data_{STOCK_SYMBOL}_{timestamp}.csv"
    df.to_csv(filename)
    
    print(f"✅ Data extracted successfully: {len(df)} records")
    print(f"📁 Saved to: {filename}")
    print(f"📊 Date range: {df.index.min()} to {df.index.max()}")
    
    return filename

if __name__ == "__main__":
    extract_stock_data()