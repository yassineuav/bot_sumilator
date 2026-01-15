import os
import sys
import pandas as pd

# Add core to path
sys.path.append(os.path.join(os.getcwd(), 'core'))

import data_loader
import features
import patterns
import model

def test_training():
    symbol = 'SPY'
    interval = '15m'
    
    print(f"Testing training for {symbol} {interval}...")
    
    # 1. Fetch and process data
    df = data_loader.fetch_data(symbol, interval=interval, period='60d')
    if df.empty:
        print("No data found")
        return
        
    df = features.compute_features(df)
    df = patterns.label_data(df)
    
    # 2. Train and Save
    tm = model.TradingModel(symbol=symbol, interval=interval)
    tm.train(df)
    
    # 3. Verify file exists
    expected_path = os.path.join('trained_models', f"model_{symbol}_{interval}.joblib")
    if os.path.exists(expected_path):
        print(f"SUCCESS: Model saved to {expected_path}")
    else:
        print(f"FAILURE: Model NOT saved to {expected_path}")

if __name__ == "__main__":
    test_training()
