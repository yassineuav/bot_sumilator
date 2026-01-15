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
    
    # Bullish Trend Start Criteria:
    # - RSI > 50 and rising
    # - MACD bullish crossover (line > signal)
    # - Price > MA20 & Price > MA50
    # - Higher High (using previous High)
    bullish_mask = (
        (df['rsi'] > 50) & 
        (df['rsi'] > df['rsi'].shift(1)) &
        (df['macd_line'] > df['macd_signal']) &
        (df['Close'] > df['ma20']) &
        (df['Close'] > df['ma50']) &
        (df['High'] > df['High'].shift(1))
    )
    
    # Bearish Trend Start Criteria:
    # - RSI < 50 and falling
    # - MACD bearish crossover (line < signal)
    # - Price < MA20 & Price < MA50
    # - Lower Low
    bearish_mask = (
        (df['rsi'] < 50) & 
        (df['rsi'] < df['rsi'].shift(1)) &
        (df['macd_line'] < df['macd_signal']) &
        (df['Close'] < df['ma20']) &
        (df['Close'] < df['ma50']) &
        (df['Low'] < df['Low'].shift(1))
    )
    
    df['label'] = 0
    df.loc[bullish_mask, 'label'] = 1
    df.loc[bearish_mask, 'label'] = -1
    
    # Future target (1 if price moves in our direction in next N bars)
    # This is useful for training. Let's look 5 bars ahead.
    horizon = 5
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_return'] = (df['future_close'] - df['Close']) / df['Close']
    
    # Refine label: It's only a good "1" if price actually goes up in future
    # and only a good "-1" if price actually goes down.
    # We use this as the training target.
    df['target'] = 0
    df.loc[(df['label'] == 1) & (df['future_return'] > 0.001), 'target'] = 1
    df.loc[(df['label'] == -1) & (df['future_return'] < -0.001), 'target'] = -1
    
    return df.dropna()

if __name__ == "__main__":
    import data_loader
    import features
    df = data_loader.fetch_data('SPY', interval='5m', period='30d')
    features_df = features.compute_features(df)
    labeled_df = label_data(features_df)
    print(labeled_df['target'].value_counts())
