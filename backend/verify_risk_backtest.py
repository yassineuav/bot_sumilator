
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add core to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from backtest import Backtester
from risk import RiskManager

def test_risk_backtest_direct():
    print("Testing LSTM Backtest Class Directly...")
    
    # 1. Mock Signals DataFrame
    # Create 100 periods of data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({
        'Datetime': dates,
        'Close': np.linspace(100, 110, 100), # Upward trend
        'signal': ['NO TRADE'] * 100,
        'confidence': [0.0] * 100
    })
    
    # Inject a Signal
    # Index 10: CALL
    df.loc[10, 'signal'] = 'CALL'
    df.loc[10, 'confidence'] = 0.9
    
    # Index 20: PUT
    df.loc[20, 'signal'] = 'PUT' 
    df.loc[20, 'confidence'] = 0.8
    
    # 2. Initialize Backtester with Custom Params
    initial_balance = 5000
    risk_pct = 0.10 # 10%
    stop_loss = 0.05 # 5%
    take_profit = 0.20 # 20%
    
    bt = Backtester(
        signals_df=df,
        symbol="TEST",
        initial_balance=initial_balance,
        interval="5m",
        target_dte=0.0,
        risk_per_trade_pct=risk_pct,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit
    )
    
    # 3. Verify Params propagated
    print("Verifying Risk Manager Parameters...")
    assert bt.risk_manager.balance == initial_balance
    assert bt.risk_manager.risk_per_trade_pct == risk_pct
    assert bt.risk_manager.stop_loss_pct == stop_loss
    assert bt.risk_manager.take_profit_pct == take_profit
    
    print("✅ Risk Parameters Propagated Successfully")
    
    # 4. Run Backtest (Mock run)
    # We expect trades to be generated.
    # Note: simulate_contract_price in Backtest -> OptionsEngine might need fetching or be mocked.
    # OptionsEngine tries to get risk_free_rate etc.
    # To avoid external calls, we might hit issues if OptionsEngine hits API.
    # BUT OptionsEngine.simulate_contract_price uses BlackScholes formula, usually no API calls unless it fetches treasury yield.
    # Let's try running.
    
    try:
        results = bt.run()
        print("Backtest Run Completed.")
        print(bt.journal.get_summary())
        
        # Check first trade size
        if bt.journal.trades:
            t1 = bt.journal.trades[0]
            expected_size = initial_balance * risk_pct
            print(f"Trade 1 Size: {t1['size']} (Expected ~{expected_size})")
            
            # Allow small diff? No, logic is exact: calculate_position_size = balance * risk_pct
            # But balance changes if we start loop. The first trade is at index 10.
            # Balance should be initial still.
            # Wait, calculate_position_size is called at entry.
            if abs(t1['size'] - expected_size) < 1.0:
                 print("✅ Trade size valid")
            else:
                 print(f"❌ Trade size invalid: Got {t1['size']}, Expected {expected_size}")
        else:
            print("⚠️ No trades generated. Check signal logic.")
            
    except Exception as e:
        print(f"❌ Backtest execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_risk_backtest_direct()
