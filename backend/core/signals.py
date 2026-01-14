import pandas as pd
import numpy as np

def generate_signals(df, model):
    """
    Generates trading signals based on model predictions.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # Get predictions and probabilities
    preds = model.predict(df)
    probs = model.predict_proba(df)
    
    df['prediction'] = preds
    # Probability of the predicted class
    df['confidence'] = np.max(probs, axis=1)
    
    # Define signal type with threshold
    df['signal'] = 'NO TRADE'
    
    # Only assign signals if confidence is >= 0.55
    df.loc[(df['prediction'] == 1) & (df['confidence'] >= 0.55), 'signal'] = 'CALL'
    df.loc[(df['prediction'] == -1) & (df['confidence'] >= 0.55), 'signal'] = 'PUT'
    
    # Add reasoning (Top features or indicator status)
    df['reason'] = ""
    for i, row in df.iterrows():
        reasons = []
        if row['signal'] == 'CALL':
            if row['rsi'] > 50: reasons.append("RSI Bullish")
            if row['macd_diff'] > 0: reasons.append("MACD Cross")
            if row['Close'] > row['ma20']: reasons.append("Above MA20")
        elif row['signal'] == 'PUT':
            if row['rsi'] < 50: reasons.append("RSI Bearish")
            if row['macd_diff'] < 0: reasons.append("MACD Cross")
            if row['Close'] < row['ma20']: reasons.append("Below MA20")
        df.at[i, 'reason'] = ", ".join(reasons)
        
    return df

if __name__ == "__main__":
    import data_loader
    import features
    import model
    
    tm = model.TradingModel()
    if tm.load():
        df = data_loader.fetch_data('SPY', interval='15m', period='5d')
        df = features.compute_features(df)
        signals_df = generate_signals(df, tm)
        print(signals_df[['Datetime', 'signal', 'confidence', 'reason']].tail(10))
    else:
        print("Model not found. Please train first.")
