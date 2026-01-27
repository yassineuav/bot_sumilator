import requests
import time
import sys

BASE_URL = "http://localhost:8000/api"

def test_api():
    print("1. Testing LSTM Training Start...")
    try:
        # Start training (short epochs for test)
        payload = {
            "symbol": "SPY", 
            "interval": "5m", 
            "epochs": 1, 
            "batch_size": 32,
            "lookback": 10,
            "period": "1mo" # Fast
        }
        res = requests.post(f"{BASE_URL}/train/lstm/", json=payload, timeout=30)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.json()}")
        
        # We need to wait for training to finish so a version is created
        print("Waiting for training to complete (approx 5s)...")
        time.sleep(5) 

        print("\n2. Testing Version List...")
        res = requests.get(f"{BASE_URL}/train/lstm/versions/", params={"symbol": "SPY", "interval": "5m"})
        data = res.json()
        print(f"Versions found: {len(data.get('versions', []))}")
        if data.get('versions'):
            latest_version = data['versions'][0]['version_id']
            print(f"Latest Version ID: {latest_version}")
            
            print("\n3. Testing Rollback/Activate...")
            rb_payload = {
                "symbol": "SPY",
                "interval": "5m",
                "timestamp": latest_version
            }
            res = requests.post(f"{BASE_URL}/train/lstm/rollback/", json=rb_payload)
            print(f"Rollback Status: {res.status_code}")
            print(f"Response: {res.json()}")
        
        print("\n4. Testing LSTM Backtest...")
        bt_payload = {
            "symbol": "SPY",
            "interval": "5m",
            "period": "5d", # Short period for test
            "model_type": "lstm"
        }
        # Backtest might calculate features on fly, might be slow
        res = requests.post(f"{BASE_URL}/backtest/", json=bt_payload, timeout=10)
        print(f"Backtest Status: {res.status_code}")
        # Print snippet of result
        print(f"Response Keys: {res.json().keys()}")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        print("Ensure Django server is running on port 8000")

if __name__ == "__main__":
    test_api()
