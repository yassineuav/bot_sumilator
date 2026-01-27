import pandas as pd
import numpy as np
import os
import joblib
import json
import traceback
from datetime import datetime

# Disable GPU/OneDNN to prevent hanging
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from .data_loader import fetch_data
    from .features import compute_features
except ImportError:
    # Allow running as script
    from data_loader import fetch_data
    from features import compute_features


MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models', 'cascade')

if not os.path.exists(MODEL_BASE_DIR):
    os.makedirs(MODEL_BASE_DIR)

class TrendModel:
    """
    Model A: Trend Filter (1D avg daily candles)
    Purpose: "Should I look for CALLs or PUTs?"
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.interval = '1d'
        self.model = None
        self.model_path = os.path.join(MODEL_BASE_DIR, symbol, 'trend_model.joblib')
        
    def train(self, df):
        print(f"--- Training TrendModel (A) for {self.symbol} ---")
        # Target: Next Day Close > MA200 ? Or simple Up/Down?
        # Prompt: Class -1 (Below MA200), 0 (Neutral), +1 (Above MA200)
        # Simplified for now: 1 (Bullish), -1 (Bearish) based on future return?
        # Or Regime classification?
        
        # Let's use:
        # 1 = Close(T+1) > Close(T) AND Close(T) > MA200 (Strong Bull)
        # -1 = Close(T+1) < Close(T) AND Close(T) < MA200 (Strong Bear)
        # 0 = Choppy / Mixed
        
        df = df.copy()
        
        # Labeling
        df['future_close'] = df['Close'].shift(-1)
        df['ma200'] = df['ma200'] # From features
        
        conditions = [
            (df['future_close'] > df['Close']) & (df['Close'] > df['ma200']),
            (df['future_close'] < df['Close']) & (df['Close'] < df['ma200'])
        ]
        choices = [1, -1]
        df['target'] = np.select(conditions, choices, default=0)
        
        # Features
        feature_cols = ['rsi', 'macd_line', 'macd_signal', 'adx', 'dist_ma50', 'dist_ma200', 'volatility_regime']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        df = df.dropna()
        
        X = df[feature_cols]
        y = df['target']
        
        if len(X) < 100:
             print("Warning: Not enough data for TrendModel")
             return
             
        # Train RandomForest (lightweight)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.model.fit(X, y)
        
        # Save
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        joblib.dump(self.model, self.model_path)
        print("TrendModel Traing Complete.")
        
    def predict(self, df):
        if self.model is None:
            if not self.load(): return 0 # Default Neutral
            
        feature_cols = ['rsi', 'macd_line', 'macd_signal', 'adx', 'dist_ma50', 'dist_ma200', 'volatility_regime']
        X = df[feature_cols].iloc[[-1]] # Last row
        
        try:
            pred = self.model.predict(X)[0]
            return pred
        except:
            return 0

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

class StructureModel:
    """
    Model B: Intraday Structure (1H)
    Purpose: "Continuation vs Reversal?"
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.interval = '1h'
        self.model = None
        self.model_path = os.path.join(MODEL_BASE_DIR, symbol, 'structure_model.joblib')
        
    def train(self, df):
        print(f"--- Training StructureModel (B) for {self.symbol} ---")
        # Target:
        # 1 = Continuation (Next 3 candles continue trend)
        # 0 = Range/Reversal
        
        df = df.copy()
        df['future_return_3h'] = df['Close'].shift(-3) / df['Close'] - 1
        
        # Define Structure Labels
        # If trend is UP (RSI>50) and Future Return > 0.2% -> Continuation (1)
        # If trend is DOWN (RSI<50) and Future Return < -0.2% -> Continuation (1)
        # Else 0
        
        conditions = [
            (df['rsi'] > 50) & (df['future_return_3h'] > 0.002),
            (df['rsi'] < 50) & (df['future_return_3h'] < -0.002)
        ]
        df['target'] = np.select(conditions, [1, 1], default=0) # Binary for simplicity: Pro-Trend vs Chop
        
        feature_cols = ['rsi', 'macd_diff', 'adx', 'vol_delta', 'higher_high', 'lower_low', 'vwap_dist']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        df = df.dropna()
        X = df[feature_cols]
        y = df['target']
        
        if len(X) < 100: return
        
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        self.model.fit(X, y)
        
        if not os.path.exists(os.path.dirname(self.model_path)):
             os.makedirs(os.path.dirname(self.model_path))
        joblib.dump(self.model, self.model_path)
        print("StructureModel Training Complete.")

    def predict(self, df):
        if self.model is None:
            if not self.load(): return 0
            
        feature_cols = ['rsi', 'macd_diff', 'adx', 'vol_delta', 'higher_high', 'lower_low', 'vwap_dist']
        if df.empty: return 0
        X = df[feature_cols].iloc[[-1]]
        try:
             return self.model.predict(X)[0]
        except:
             return 0

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

class ExecutionModel:
    """
    Model C: Execution (15m LSTM)
    Purpose: "Enter NOW?"
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.interval = '15m'
        self.lookback = 60 # 15 hours approx
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.base_dir = os.path.join(MODEL_BASE_DIR, symbol, 'execution_lstm')
        if not os.path.exists(self.base_dir): os.makedirs(self.base_dir)
        
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax')) # UP, NEUTRAL, DOWN
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, df):
        print(f"--- Training ExecutionModel (C) for {self.symbol} ---")
        
        # Target: Next Candle Direction
        # UP (0), NEUTRAL (1), DOWN (2) ? 
        # Usually classifiers map 0..N.
        # Let's map: 0=UP, 1=NEUTRAL, 2=DOWN (Standard Keras to_categorical)
        
        df = df.copy()
        df['ret'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # Thresholds
        threshold = 0.0005 # 0.05% move required to be directional (small for 15m)
        
        conditions = [
            (df['ret'] > threshold),
            (df['ret'] < -threshold)
        ]
        choices = [0, 2] # 0=UP, 2=DOWN
        df['target'] = np.select(conditions, choices, default=1) # 1=NEUTRAL
        
        # Features: OHLCV + Indicators + Context from A/B (passed in df usually? Or we blindly use technicals)
        # Using pure technicals here first.
        feature_cols = ['rsi', 'macd_line', 'vwap_dist', 'vol_delta', 'atr', 'volatility_regime']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        df = df.dropna()
        
        data = df[feature_cols].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.lookback):
            X.append(scaled_data[i:i+self.lookback])
            y.append(df['target'].iloc[i+self.lookback])
            
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 100: return
        
        # One-hot encoding y
        y = tf.keras.utils.to_categorical(y, num_classes=3)
        
        self.build_model((X.shape[1], X.shape[2]))
        
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Save
        self.model.save(os.path.join(self.base_dir, 'model.keras'))
        joblib.dump(self.scaler, os.path.join(self.base_dir, 'scaler.joblib'))
        print("ExecutionModel Training Complete.")

    def predict_proba(self, df):
        # Return [Prob_UP, Prob_NEUTRAL, Prob_DOWN]
        if self.model is None:
            if not self.load(): return [0, 1, 0]
            
        feature_cols = ['rsi', 'macd_line', 'vwap_dist', 'vol_delta', 'atr', 'volatility_regime']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        data = df[feature_cols].tail(self.lookback).values
        if len(data) < self.lookback: return [0, 1, 0]
        
        scaled_data = self.scaler.transform(data)
        X = scaled_data.reshape(1, self.lookback, len(feature_cols))
        
        probs = self.model.predict(X, verbose=0)[0] # [p0, p1, p2] -> [UP, NEUTRAL, DOWN]
        return probs

    def load(self):
        model_p = os.path.join(self.base_dir, 'model.keras')
        scaler_p = os.path.join(self.base_dir, 'scaler.joblib')
        if os.path.exists(model_p) and os.path.exists(scaler_p):
            self.model = load_model(model_p)
            self.scaler = joblib.load(scaler_p)
            return True
        return False

class CascadeSystem:
    def __init__(self, symbol='SPY'):
        self.symbol = symbol
        self.trend_model = TrendModel(symbol)
        self.structure_model = StructureModel(symbol)
        self.exec_model = ExecutionModel(symbol)
        
    def train_system(self):
        print("=== Starting Cascade System Training ===")
        
        # 1. Trend Model (1D)
        df_1d = fetch_data(self.symbol, interval='1d', period='5y')
        df_1d = compute_features(df_1d)
        self.trend_model.train(df_1d)
        
        # 2. Structure Model (1H)
        df_1h = fetch_data(self.symbol, interval='1h', period='2y')
        df_1h = compute_features(df_1h)
        self.structure_model.train(df_1h)
        
        # 3. Execution Model (15m)
        df_15m = fetch_data(self.symbol, interval='15m', period='60d')
        df_15m = compute_features(df_15m)
        self.exec_model.train(df_15m)
        
        print("=== Cascade System Training Finished ===")
        
    def get_signal(self):
        # 1. Fetch live data context
        # Trend
        df_1d = fetch_data(self.symbol, interval='1d', period='2y') # Need enough for MA200 (approx 200 trading days)
        df_1d = compute_features(df_1d)
        trend = self.trend_model.predict(df_1d) # 1, -1, 0
        
        # Structure
        df_1h = fetch_data(self.symbol, interval='1h', period='60d')
        df_1h = compute_features(df_1h)
        structure = self.structure_model.predict(df_1h) # 1 (Continuation), 0 (Chop)
        
        # Execution
        df_15m = fetch_data(self.symbol, interval='15m', period='5d') # Need lookback
        df_15m = compute_features(df_15m)
        probs = self.exec_model.predict_proba(df_15m) # [UP, NEUTRAL, DOWN]
        
        # === CASCADE LOGIC ===
        # Gate 1: Check Trend
        # Gate 2: Check Structure
        # Gate 3: Check Execution
        
        final_signal = 'NEUTRAL'
        confidence = 0.0
        reason = "Wait"
        
        prob_up = probs[0]
        prob_neutral = probs[1]
        prob_down = probs[2]
        
        exec_signal = 'NEUTRAL'
        if prob_up > 0.5: exec_signal = 'CALL'
        elif prob_down > 0.5: exec_signal = 'PUT'
        
        # Combine
        # If Trend is Bullish (1) AND Structure is Continuation (1) AND Exec is CALL -> CALL
        if trend == 1 and structure == 1 and exec_signal == 'CALL':
            final_signal = 'CALL'
            confidence = prob_up
            reason = "Trend+Struct+Exec Align Bullish"
            
        elif trend == -1 and structure == 1 and exec_signal == 'PUT':
            final_signal = 'PUT'
            confidence = prob_down
            reason = "Trend+Struct+Exec Align Bearish"
            
        else:
            final_signal = 'NEUTRAL'
            confidence = max(probs)
            if structure == 0: reason = "Structure Chop/Range"
            elif trend == 0: reason = "Trend Neutral"
            elif trend != (1 if exec_signal=='CALL' else -1): reason = f"Trend({trend}) vs Exec({exec_signal}) Divergence"
            
        return {
            'signal': final_signal,
            'confidence': float(confidence),
            'reason': reason,
            'components': {
                'trend': int(trend),
                'structure': int(structure),
                'execution_probs': [float(p) for p in probs]
            }
        }
