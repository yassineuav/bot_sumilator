
import requests
import json
import traceback

def test_backtest_api():
    url = 'http://localhost:8000/api/backtest/'
    payload = {
        "symbol": "SPY",
        "interval": "15m",
        "initial_balance": 1000,
        "risk_pct": 20,
        "stop_loss": 10,
        "take_profit": 50,
        "project_name": "default",
        "model_type": "lstm"
    }
    
    print(f"Sending POST to {url} with payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"Status Code: {response.status_code}")
        try:
            data = response.json()
            # print formatted json
            print("Response JSON:")
            print(json.dumps(data, indent=2))
        except:
            print("Response Text (Not JSON):")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_backtest_api()
