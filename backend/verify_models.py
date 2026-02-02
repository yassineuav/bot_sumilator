import os
import sys
import django
import pandas as pd
from datetime import datetime

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
    results = {}
    
    print("="*60)
    print(f" MODEL VERIFICATION RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Symbol: {symbol} | Interval: {interval}")
    print("="*60)
    
    # 1. Fetch Data
    print("\n[1/4] Fetching Data...")
    try:
        pipeline = MultiTimeframePipeline(symbol)
        df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period='60d')
        if df.empty:
            print("❌ Error: No data fetched.")
            return
        print(f"✅ Data fetched: {len(df)} rows")
    except Exception as e:
        print(f"❌ Data Fetch Error: {e}")
        return

    # 2. Verify XGBoost
    print("\n[2/4] Testing XGBoost...")
    try:
        df_labeled = patterns.label_data(df)
        xgb = TradingModel(symbol, interval)
        xgb.train(df_labeled)
        probs = xgb.predict_proba(df.iloc[[-1]])
        print(f"✅ XGB Training Success. Prediction Probs: {probs[0]}")
        results['XGBoost'] = "SUCCESS"
    except Exception as e:
        print(f"❌ XGB Error: {e}")
        results['XGBoost'] = f"FAILED: {str(e)[:50]}"

    # 3. Verify LSTM
    print("\n[3/4] Testing LSTM...")
    try:
        lstm = LSTMModel(symbol, interval)
        lstm.train(df, epochs=2) # Short train for test
        next_price = lstm.predict_next(df)
        current_close = df['Close'].iloc[-1]
        diff = next_price - current_close
        print(f"✅ LSTM Success. Next Price: {next_price:.2f} (Current: {current_close:.2f}, Diff: {diff:.2f})")
        results['LSTM'] = "SUCCESS"
    except Exception as e:
        print(f"❌ LSTM Error: {e}")
        results['LSTM'] = f"FAILED: {str(e)[:50]}"

    # 4. Verify Hybrid
    print("\n[4/4] Testing Hybrid...")
    try:
        hm = HybridModel(symbol, interval)
        # Should reuse trained models if loaded
        hm.train(df_labeled) 
        pred = hm.get_prediction_now(df)
        print(f"✅ Hybrid Success. Prediction: {pred['signal']} (Conf: {pred['confidence']:.2f})")
        results['Hybrid'] = "SUCCESS"
    except Exception as e:
        print(f"❌ Hybrid Error: {e}")
        results['Hybrid'] = f"FAILED: {str(e)[:50]}"

    print("\n" + "="*60)
    print(" VERIFICATION SUMMARY")
    print("-" * 60)
    for model, status in results.items():
        print(f" {model:<15} : {status}")
    print("="*60)

if __name__ == "__main__":
    verify_models()
