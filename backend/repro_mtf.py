import os
import django
import sys
import threading
import time

# Setup Django Environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_backend.settings')
django.setup()

from core.feature_pipeline import MultiTimeframePipeline
from core.lstm_model import LSTMModel

def run_repro():
    symbol = 'SPY'
    interval = '15m'
    period = '60d'
    
    print(f"--- Starting Repro: MTF Training for {symbol} {interval} ---")
    try:
        pipeline = MultiTimeframePipeline(symbol)
        print("Pipeline initialized.")
        
        print("Preparing data...")
        # This is where the log said it failed before
        df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period=period)
        print(f"Data prepared. Shape: {df.shape}")
        
        if df.empty:
            print("ERROR: Data is empty.")
            return

        print("Initializing LSTM...")
        lstm = LSTMModel(symbol, interval, lookback=60)
        
        print("Training LSTM...")
        lstm.train(df, epochs=1, batch_size=32)
        print("Training complete.")
        
    except Exception as e:
        print(f"CAUGHT EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_repro()
