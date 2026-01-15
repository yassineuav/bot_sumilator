import os
import sys
import pandas as pd
from datetime import datetime

# Add core to path
sys.path.append(os.path.join(os.getcwd(), 'core'))

import data_loader
import features
import patterns
import model
from manual_tester import ManualTester

def test_manual():
    symbol = 'SPY'
    interval = '15m'
    
    # Get a valid historical timestamp from data_loader
    df = data_loader.fetch_data(symbol, interval=interval, period='5d')
    if df.empty:
        print("No data found")
        return
        
    ts = df['Datetime'].iloc[-5] # Use a timestamp slightly in the past
    print(f"Testing manual prediction at {ts}...")
    
    try:
        tester = ManualTester(symbol, interval)
        pred = tester.run_prediction_at_time(ts)
        print("Prediction Result:", pred)
        
        if pred:
            option_data = tester.select_otm_option(pred['entry_price'], pred['signal'], ts)
            print("Option Data:", option_data)
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_manual()
