import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Ensure we can import core modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data_loader
import features
import model
from options_engine import OptionsEngine

class ManualTester:
    def __init__(self, symbol, interval, model_type='xgb'):
        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type
        self.options_engine = OptionsEngine()
        
        # Load appropriate model
        if model_type == 'lstm':
            from core.lstm_model import LSTMModel
            self.tm = LSTMModel(symbol=symbol, interval=interval)
        elif model_type == 'hybrid':
            from core.hybrid_model import HybridModel
            self.tm = HybridModel(symbol=symbol, interval=interval)
        else:
            # Default to XGBoost
            self.tm = model.TradingModel(symbol=symbol, interval=interval)
            
        self.tm.load()

    def run_prediction_at_time(self, timestamp):
        """
        Runs the predictor on historical data up to the given timestamp.
        """
        # 1. Fetch data using Pipeline (handles Multi-Timeframe features)
        from core.feature_pipeline import MultiTimeframePipeline
        
        # Determine period logic similar to views.py
        if self.interval == '1d':
            period = '10y'
        elif self.interval in ['1h', '90m']:
            period = '730d'
        else:
            period = '60d'
            
        pipeline = MultiTimeframePipeline(self.symbol)
        try:
            # prepare_multitimeframe_data fetches and computes ALL features
            df = pipeline.prepare_multitimeframe_data(base_interval=self.interval, base_period=period)
        except Exception as e:
            print(f"ManualTester Pipeline Error: {e}")
            return None

        if df.empty:
            return None

        # 2. Filter data up to the selected timestamp
        # Ensure Datetime column is datetime objects
        # MultiTimeframePipeline returns DF with Datetime Index? 
        # Let's check: prepare_multitimeframe_data returns base_df which might be indexed by Datetime.
        # We need 'Datetime' as a column for filtering logic below or reset it.
        
        if 'Datetime' not in df.columns:
            df = df.reset_index()
            # If still missing, try rename
            if 'Datetime' not in df.columns:
                 if df.columns[0].lower() in ['date', 'datetime', 'index']:
                     df.rename(columns={df.columns[0]: 'Datetime'}, inplace=True)
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        ts = pd.to_datetime(timestamp)
        
        # Sync timezone awareness between column and search timestamp
        if df['Datetime'].dt.tz is not None and ts.tzinfo is None:
            # Assume naive timestamps are UTC
            ts = ts.tz_localize('UTC').tz_convert(df['Datetime'].dt.tz)
        elif df['Datetime'].dt.tz is None and ts.tzinfo is not None:
             ts = ts.replace(tzinfo=None)

        df_filtered = df[df['Datetime'] <= ts].copy()
        
        if df_filtered.empty:
            return None

        # 3. Features already computed by Pipeline
        df_features = df_filtered

        # 4. Predict on the last available candle (which is exactly at or before our timestamp)
        current_data = df_features.iloc[[-1]].copy()
        probs = self.tm.predict_proba(current_data)[0]
        pred_idx = np.argmax(probs)
        signal_map = {0: 'PUT', 1: 'NEUTRAL', 2: 'CALL'}
        
        signal = signal_map[pred_idx]
        confidence = float(probs[pred_idx])
        entry_price = float(current_data['Close'].iloc[0])

        return {
            'signal': signal,
            'confidence': confidence,
            'probs': {
                'bearish': float(probs[0]),
                'neutral': float(probs[1]),
                'bullish': float(probs[2])
            },
            'entry_price': entry_price,
            'timestamp': str(current_data['Datetime'].iloc[-1])
        }

    def select_otm_option(self, spot_price, signal, timestamp):
        """
        Selects an OTM option based on premium range $0.10 - $0.30
        """
        if signal not in ['CALL', 'PUT']:
            return None

        # Strike selection logic (OTM)
        # Using simple offsets for simulation since we don't have a real options chain here
        # Premium range $0.10 - $0.30
        
        # We'll use Black-Scholes to find a strike that fits the premium range
        # Assumptions: r=0.05, vol from recent history, T=0.5 (0DTE/1DTE approx)
        
        # Estimate volatility
        # (Simplified estimate for manual test)
        vol = 0.20 
        T = 0.5 / 365.0 # Very short dated
        r = 0.05
        
        option_type = 'call' if signal == 'CALL' else 'put'
        
        # Try a few strikes to find one in the premium range
        # SPY moves in ~$1 increments for strikes
        base_strike = round(spot_price)
        offsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        found_option = None
        
        # 2. Iterate to find best strike in range
        for offset in offsets:
            strike = base_strike + offset if signal == 'CALL' else base_strike - offset
            
            # Filter unrealistic strikes (e.g. negative)
            if strike <= 0: continue
            
            res = self.options_engine.black_scholes(
                S=spot_price, K=strike, T=T, r=r, sigma=vol, option_type=option_type
            )
            price = res['price']
            
            # Logic: Match desired price range ($0.10 - $0.50 for cheap/lotto options)
            # OR just return the first one that is "Reasonable" OTM
            
            if 0.10 <= price <= 0.60:
                found_option = {
                    'strike': strike,
                    'premium': round(price, 2),
                    'type': signal,
                    'expiry': '0DTE (Simulated)', 
                    'delta': res['delta'],
                    'theta': res['theta_daily']
                }
                break
        
        # Fallback: Just take offset 3 ( ~$2-$3 OTM usually)
        if not found_option:
             strike = base_strike + 3 if signal == 'CALL' else base_strike - 3
             if strike > 0:
                 res = self.options_engine.black_scholes(S=spot_price, K=strike, T=T, r=r, sigma=vol, option_type=option_type)
                 found_option = {
                    'strike': strike,
                    'premium': round(res['price'], 2),
                    'type': signal,
                    'expiry': '0DTE (Simulated)',
                    'delta': res['delta'],
                    'theta': res['theta_daily']
                }

        return found_option
