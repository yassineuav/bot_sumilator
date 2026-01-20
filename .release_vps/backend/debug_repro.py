
import sys
import os
import pandas as pd

# Path setup
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from backtest import Backtester
from risk import RiskManager

def test_repro():
    print("Starting Repro...")
    
    # Mock settings
    take_profit = 5.0 # 500%
    risk_pct = 0.2
    stop_loss = 0.1
    
    # Mock DataFrame
    dates = pd.date_range('2025-01-01', periods=10, freq='1h')
    df = pd.DataFrame({
        'Datetime': dates,
        'Close': [100, 101, 102, 103, 105, 110, 105, 100, 95, 90],
        'signal': ['CALL', 'NO TRADE', 'NO TRADE', 'NO TRADE', 'NO TRADE', 'NO TRADE', 'PUT', 'NO TRADE', 'NO TRADE', 'NO TRADE'],
        'confidence': [0.9] * 10
    })
    
    bt = Backtester(df, initial_balance=1000)
    bt.risk_manager.risk_per_trade_pct = risk_pct
    bt.risk_manager.stop_loss_pct = stop_loss
    
    # The line from views.py
    tp_s = [(take_profit, 1.0)]
    print(f"Setting TP Stages to: {tp_s}")
    bt.risk_manager.tp_stages = tp_s
    
    print("Running Backtest...")
    try:
        bt.run()
        print("Success!")
    except Exception as e:
        print(f"Caught Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()
