import pandas as pd
import numpy as np
import ta

def compute_features(df):
    """
    Computes technical indicators as features for the ML model.
    """
    if df.empty:
        return df
    
    # Copy to avoid modifying the original dataframe
    df = df.copy()
    
    # 1. RSI (14)
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    # 2. MACD (12, 26, 9)
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 3. Moving Averages
    df['ma20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['ma50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['ma200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # 4. Volume Features
    df['vol_ma20'] = ta.trend.sma_indicator(df['Volume'], window=20)
    df['vol_spike'] = df['Volume'] / df['vol_ma20']
    
    # 5. Price Action Features
    # Returns
    df['returns'] = df['Close'].pct_change()
    
    # Volatility
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Momentum Expansion
    df['momentum'] = df['Close'] - df['Close'].shift(5)
    
    # Distance from MAs
    df['dist_ma20'] = (df['Close'] - df['ma20']) / df['ma20']
    df['dist_ma50'] = (df['Close'] - df['ma50']) / df['ma50']
    
    # 6. Trend Strength (ADX)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # 7. Higher Highs / Lower Lows (simple version)
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    
    # Clean up NaNs created by indicators
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    import data_loader
    df = data_loader.fetch_data('SPY', interval='5m', period='5d')
    features_df = compute_features(df)
    print(features_df.head())
    print(f"Features computed: {features_df.columns.tolist()}")
