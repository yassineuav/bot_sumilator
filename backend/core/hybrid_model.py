import pandas as pd
import numpy as np
import os
import joblib
from .model import TradingModel
from .lstm_model import LSTMModel

class HybridModel:
    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.xgb_model = TradingModel(symbol, interval)
        self.lstm_model = LSTMModel(symbol, interval)
        
    def train(self, df, force_retrain=False):
        # Train both models
        print(f"Training Hybrid Model for {self.symbol} {self.interval}...")
        
        # Check if we should retrain or if models are already loaded
        if not force_retrain:
            loaded = self.load()
            if loaded == True:
                print("Hybrid Model: Sub-models already loaded, skipping training.")
                return

        # 1. Train XGBoost
        print("--- Training XGBoost Sub-Model ---")
        self.xgb_model.train(df)
        
        # 2. Train LSTM
        print("--- Training LSTM Sub-Model ---")
        # LSTM might need specific data prep handled inside its train method
        # The dataframe passed here should have all necessary columns
        self.lstm_model.train(df, epochs=10) # Reduced epochs for standard training
        
        print("Hybrid Model Training Complete.")
        
    def load(self):
        # Try to load both
        xgb_loaded = self.xgb_model.load()
        lstm_loaded = self.lstm_model.load()
        
        if xgb_loaded and lstm_loaded:
            print("Hybrid Model: Both sub-models loaded successfully.")
            return True
        elif xgb_loaded:
             print("Hybrid Model Warning: Only XGBoost loaded.")
             return "PARTIAL_XGB"
        elif lstm_loaded:
             print("Hybrid Model Warning: Only LSTM loaded.")
             return "PARTIAL_LSTM"
        else:
             print("Hybrid Model: No sub-models found.")
             return False

    def predict_signals(self, df):
        """
        Generate signals where BOTH models agree.
        If One says BUY and other says SELL/NEUTRAL -> NEUTRAL.
        If Both say BUY -> CALL
        If Both say SELL -> PUT
        """
        # Get XGB predictions (DataFrame with 'signal' column? No, generate_signals does that. 
        # TradingModel.predict just gives values)
        
        # We need to use the logic from the respective "generate_signals" workflows.
        # But here we want a unified signal.
        
        # 1. XGB Signals
        # We need to re-implement or call the signal generation logic for XGB
        # Ideally, we call specific methods that return aligned arrays/series.
        
        # XGB Predict: returns mapped integers (-1, 0, 1)
        # Note: TradingModel.predict returns numpy array of -1, 0, 1
        xgb_preds = self.xgb_model.predict(df) 
        
        # LSTM Predict: returns DataFrame with 'lstm_pred' and 'signal' ('CALL', 'PUT', 'NEUTRAL')
        # We need to extract the array of mapped integers or compare signals directly.
        lstm_df_out = self.lstm_model.predict_signals(df)
        lstm_signals = lstm_df_out['signal'].values # ['CALL', 'PUT', 'NEUTRAL']
        
        # Align lengths? 
        # XGB predict uses `df[feature_cols]`, returning len(df) predictions (if no dropna).
        # LSTM predict_signals returns len(df) rows.
        
        if len(xgb_preds) != len(lstm_signals):
            print(f"Shape mismatch: XGB={len(xgb_preds)}, LSTM={len(lstm_signals)}")
            # Handle mismatch (truncate to min length?)
            min_len = min(len(xgb_preds), len(lstm_signals))
            xgb_preds = xgb_preds[-min_len:]
            lstm_signals = lstm_signals[-min_len:]
            
        final_signals = []
        
        for i in range(len(xgb_preds)):
            xgb_s = xgb_preds[i] # -1, 0, 1
            lstm_s = lstm_signals[i] # 'PUT', 'NEUTRAL', 'CALL'
            
            # Map XGB to String
            if xgb_s == 1: xgb_str = 'CALL'
            elif xgb_s == -1: xgb_str = 'PUT'
            else: xgb_str = 'NEUTRAL'
            
            # Consensus Logic
            # STRICT: Agree exactly
            if xgb_str == lstm_s:
                final_signals.append(xgb_str)
            else:
                final_signals.append('NEUTRAL')
                
        # Return valid signals series/array
        return final_signals

    def get_prediction_now(self, df):
        """
        Get live prediction for the latest candle.
        Returns: { 'signal': 'CALL'/'PUT'/'NEUTRAL', 'confidence': ... }
        """
        # 1. XGB
        # predict_proba returns [Bearish, Neutral, Bullish]
        xgb_probs = self.xgb_model.predict_proba(df.iloc[[-1]])[0]
        xgb_bearish, xgb_neutral, xgb_bullish = xgb_probs
        
        if xgb_bullish > 0.6: xgb_sig = 'CALL'
        elif xgb_bearish > 0.6: xgb_sig = 'PUT'
        else: xgb_sig = 'NEUTRAL'
        
        # 2. LSTM
        # predict_next returns next price
        # We need to compare it to current close
        current_close = df['Close'].iloc[-1]
        next_price = self.lstm_model.predict_next(df)
        
        lstm_sig = 'NEUTRAL'
        lstm_reldiff = (next_price - current_close) / current_close
        
        # Thresholds from plan (generic)
        if lstm_reldiff > 0.001: # 0.1% move up
            lstm_sig = 'CALL'
        elif lstm_reldiff < -0.001:
            lstm_sig = 'PUT'
            
        print(f"Hybrid Debug: XGB={xgb_sig} ({xgb_probs}), LSTM={lstm_sig} (Diff={lstm_reldiff:.4f})")
        
        # Voting
        final_sig = 'NEUTRAL'
        confidence = 0.0
        
        if xgb_sig == lstm_sig and xgb_sig != 'NEUTRAL':
            final_sig = xgb_sig
            # Avg confidence ?
            # LSTM "confidence" is vague, use XGB prob
            if final_sig == 'CALL': confidence = float(xgb_bullish)
            else: confidence = float(xgb_bearish)
        
        return {
            'signal': final_sig,
            'confidence': confidence,
            'models': {
                'xgb': {
                    'signal': xgb_sig,
                    'probs': [float(p) for p in xgb_probs]
                },
                'lstm': {
                    'signal': lstm_sig,
                    'next_price': float(next_price),
                    'rel_diff': float(lstm_reldiff)
                }
            }
        }
