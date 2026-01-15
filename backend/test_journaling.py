import os
import sys
import pandas as pd

# Add core to path
sys.path.append(os.path.join(os.getcwd(), 'core'))

import data_loader
import features
import patterns
import model
import signals
import backtest

def test_journaling():
    symbol = 'SPY'
    interval = '15m'
    
    print(f"Testing journaling for {symbol} {interval}...")
    
    # 1. Get Data
    df = data_loader.fetch_data(symbol, interval=interval, period='60d')
    df = features.compute_features(df)
    df = patterns.label_data(df)
    
    # 2. Generate Signals
    tm = model.TradingModel(symbol=symbol, interval=interval)
    if not tm.load():
        tm.train(df)
    
    signals_df = signals.generate_signals(df, tm)
    
    # 3. Run Backtest
    bt = backtest.Backtester(signals_df, symbol=symbol, interval=interval, initial_balance=1000)
    bt.run()
    
    # 4. Verify file exists
    expected_path = os.path.join('trade_journals', f"journal_{symbol}_{interval}.csv")
    if os.path.exists(expected_path):
        print(f"SUCCESS: Journal saved to {expected_path}")
        # Check content
        j_df = pd.read_csv(expected_path)
        print(f"Trades in journal: {len(j_df)}")
    else:
        print(f"FAILURE: Journal NOT saved to {expected_path}")

if __name__ == "__main__":
    test_journaling()
