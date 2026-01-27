import os
import sys
import django
import pandas as pd

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_backend.settings')
django.setup()

from core.model import TradingModel
from core.lstm_model import LSTMModel
from core.hybrid_model import HybridModel
from core.feature_pipeline import MultiTimeframePipeline
from core import patterns

def verify_models():
    symbol = 'SPY'
    interval = '15m'
    
    print(f"--- Verifying Models for {symbol} {interval} ---")
    
    # 1. Fetch Data
    pipeline = MultiTimeframePipeline(symbol)
    df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period='60d')
    if df.empty:
        print("Error: No data fetched.")
        return
        
    print(f"Data fetched: {len(df)} rows")
    
    # 2. Verify XGBoost
    print("\n[Testing XGBoost]")
    try:
        df_labeled = patterns.label_data(df)
        xgb = TradingModel(symbol, interval)
        xgb.train(df_labeled)
        probs = xgb.predict_proba(df.iloc[[-1]])
        print(f"XGB Training Success. Prediction Probs: {probs}")
    except Exception as e:
        print(f"XGB Error: {e}")
        import traceback
        traceback.print_exc()

    # 3. Verify LSTM
    print("\n[Testing LSTM]")
    try:
        lstm = LSTMModel(symbol, interval)
        lstm.train(df, epochs=2) # Short train for test
        next_price = lstm.predict_next(df)
        print(f"LSTM Training Success. Next Price Pred: {next_price:.2f} (Current: {df['Close'].iloc[-1]:.2f})")
    except Exception as e:
        print(f"LSTM Error: {e}")
        import traceback
        traceback.print_exc()

    # 4. Verify Hybrid
    print("\n[Testing Hybrid]")
    try:
        hm = HybridModel(symbol, interval)
        # Should reuse trained models if loaded, or we can retrain
        hm.train(df_labeled) 
        pred = hm.get_prediction_now(df)
        print(f"Hybrid Training Success. Prediction: {pred}")
    except Exception as e:
        print(f"Hybrid Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_models()
