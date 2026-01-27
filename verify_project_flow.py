
import requests
import json
import os
import shutil

BASE_URL = "http://127.0.0.1:8000/api"

def test_project_flow():
    project_name = "verify_proj_001"
    symbol = "SPY"
    
    print(f"--- Starting Verification for Project: {project_name} ---")
    
    # Clean up previous run if exists
    # (Optional, but good for idempotency)
    
    # 1. Train 15m Model
    print("\n1. Training 15m Model...")
    payload = {
        "symbol": symbol,
        "intervals": ["15m"],
        "model_type": "lstm",
        "project_name": project_name
    }
    try:
        r = requests.post(f"{BASE_URL}/train/", json=payload)
        print(f"Status: {r.status_code}")
        # print(f"Response: {r.json()}") # Might fail if 500
        try:
             print(f"Response: {r.json()}")
        except:
             print(f"Response Text: {r.text}")
             
        if r.status_code != 200:
            print("FAILED: Training 15m")
            return
    except Exception as e:
        print(f"FAILED: Connection error {e}")
        return

    # 2. Train 1h Model
    print("\n2. Training 1h Model...")
    payload["intervals"] = ["1h"]
    try:
        r = requests.post(f"{BASE_URL}/train/", json=payload)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.json()}")
    except Exception as e:
         print(f"FAILED: Connection error {e}")

    # 3. Verify Model Files Existence
    # Path logic logic from lstm_model.py: trained_models/lstm/{project_name}/{symbol}/{interval}
    base_path = os.path.join(os.getcwd(), 'backend', 'trained_models', 'lstm', project_name, symbol)
    
    if os.path.exists(os.path.join(base_path, '15m', 'active_version.json')):
        print("SUCCESS: 15m Model created.")
    else:
        print(f"FAILURE: 15m Model not found at {base_path}/15m")
        
    if os.path.exists(os.path.join(base_path, '1h', 'active_version.json')):
        print("SUCCESS: 1h Model created.")
    else:
        print(f"FAILURE: 1h Model not found at {base_path}/1h")
        
    # 4. Run Backtest (15m)
    print("\n4. Running Backtest (15m)...")
    bt_payload = {
        "symbol": symbol,
        "interval": "15m",
        "model_type": "lstm",
        "project_name": project_name,
        "initial_balance": 1000,
        "risk_pct": 20,
        "stop_loss": 10,
        "take_profit": 50
    }
    try:
        r = requests.post(f"{BASE_URL}/backtest/", json=bt_payload)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"Backtest Trades: {data.get('total_trades')}")
            print(f"Final Balance: {data.get('final_balance')}")
        else:
             print(f"Response: {r.text}")
    except Exception as e:
        print(f"FAILED: Connection error {e}")

    # 5. Verify Journal Existence
    # Path logic from journal.py: backend/trade_journals -> but based on CWD? 
    # Usually relative to where python runs (root or backend?)
    # Running from root?
    
    # Journal dir depends on where runserver is running. Assuming backend/
    journal_path = os.path.join(os.getcwd(), 'backend', 'trade_journals', project_name, f"journal_{symbol}_15m.csv")
    if os.path.exists(journal_path):
        print(f"SUCCESS: Journal found at {journal_path}")
    else:
        # Try checking just 'trade_journals' if CWD is backend
        journal_path_alt = os.path.join(os.getcwd(), 'trade_journals', project_name, f"journal_{symbol}_15m.csv")
        if os.path.exists(journal_path_alt):
             print(f"SUCCESS: Journal found at {journal_path_alt}")
        else:
             print(f"FAILURE: Journal not found. Checked {journal_path} and {journal_path_alt}")

if __name__ == "__main__":
    test_project_flow()
