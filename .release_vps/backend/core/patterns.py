import pandas as pd
import numpy as np

def label_data(df):
    """
    Labels the data for training based on trend start criteria.
    1 = Bullish Trend Start
    -1 = Bearish Trend Start
    0 = Neutral
    """
    df = df.copy()
    
    # Bullish Trend:
    # - Price > MA50 (Main trend filter)
    # - OR MACD Bullish OR RSI > 50
    bullish_mask = (df['Close'] > df['ma50']) & ((df['macd_line'] > df['macd_signal']) | (df['rsi'] > 50))
    
    # Bearish Trend:
    # - Price < MA50
    # - OR MACD Bearish OR RSI < 50
    bearish_mask = (df['Close'] < df['ma50']) & ((df['macd_line'] < df['macd_signal']) | (df['rsi'] < 50))
    
    df['label'] = 0
    df.loc[bullish_mask, 'label'] = 1
    df.loc[bearish_mask, 'label'] = -1
    
    # Future target (1 if price moves in our direction in next N bars)
    horizon = 10 # Predict further out for more stability
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
    
    # Training Target
    df['target'] = 0
    # Target 1: Currently Bullish AND price goes up further
    df.loc[(df['label'] == 1) & (df['future_return'] > 0.0005), 'target'] = 1
    # Target -1: Currently Bearish AND price goes down further
    df.loc[(df['label'] == -1) & (df['future_return'] < -0.0005), 'target'] = -1
    
    return df.dropna()

if __name__ == "__main__":
    import data_loader
    import features
    df = data_loader.fetch_data('SPY', interval='5m', period='30d')
    features_df = features.compute_features(df)
    labeled_df = label_data(features_df)
    print(labeled_df['target'].value_counts())
