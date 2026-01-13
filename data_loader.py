import yfinance as yf
import pandas as pd
import os

def fetch_data(symbol, interval='5m', period='60d'):
    """
    Fetches historical OHLCV data from yfinance.
    Note: '1m' interval only supports up to 7 days of data.
    """
    if interval == '1m' and period not in ['1d', '5d', '7d']:
        print(f"Interval '1m' only supports up to 7 days. Defaulting to 7d.")
        period = '7d'
        
    print(f"Fetching data for {symbol} with interval {interval} and period {period}...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        print(f"No data found for {symbol}.")
        return pd.DataFrame()
        
    # Clean up column names (yfinance sometimes returns multi-index or weird names)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    # Reset index to have 'Datetime' as a column
    data = data.reset_index()
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'Datetime'})
        
    return data

if __name__ == "__main__":
    # Test fetch
    df = fetch_data('SPY', interval='5m', period='5d')
    print(df.head())
    print(f"Fetched {len(df)} rows.")
