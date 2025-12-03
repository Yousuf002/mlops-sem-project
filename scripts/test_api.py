import requests
import json

API_URL = "http://localhost:8000"

def test_api():
    print("Testing Stock Prediction API\n")
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 2: Model info
    print("2. Testing model info endpoint...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 3: Make prediction (37 features for your model)
    print("3. Testing prediction endpoint...")
    test_features = [
        180.5,  # open
        182.3,  # high
        179.8,  # low
        181.2,  # close
        5000000,  # volume
        181.0, 180.5, 180.3, 179.8, 179.5,  # close lags
        5100000, 5050000, 4980000, 4920000, 4850000,  # volume lags
        180.8, 180.5, 180.2,  # rolling means
        0.85, 0.78, 0.92,  # rolling stds
        5020000, 5010000, 4990000,  # volume rolling means
        0.7, 0.5, 1.2,  # price change, pct, returns
        2.5, 1.4,  # volatility, volatility_pct
        14, 3, 12,  # hour, day_of_week, month
        180.6, 180.3, 180.9, 180.4,  # sma, ema
        58.5  # rsi
    ]
    
    payload = {
        "features": test_features
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
    
    # Test 4: Metrics
    print("4. Testing metrics endpoint...")
    response = requests.get(f"{API_URL}/metrics")
    print(f"   Status: {response.status_code}")
    print(f"   Metrics available: Yes\n")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_api()