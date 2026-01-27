import yfinance as yf
import pandas as pd
import os
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def fetch_data(symbol, interval='5m', period='60d'):
    """
    Fetches historical OHLCV data from yfinance.
    Note: '1m' interval only supports up to 7 days of data.
    """
    if interval == '1m' and period not in ['1d', '5d', '7d']:
        print(f"Interval '1m' only supports up to 7 days. Defaulting to 7d.")
        period = '7d'
        
    print(f"Fetching data for {symbol} with interval {interval} and period {period}...")
    
    # Check Cache
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{period}.pkl")
    use_cache = False
    if os.path.exists(cache_file):
        # Check age (e.g. 15 minutes cache for intraday, 4 hours for daily)
        mtime = os.path.getmtime(cache_file)
        age = time.time() - mtime
        
        limit = 15 * 60 # Default 15 mins
        if interval in ['1d', '1wk']:
            limit = 4 * 3600 # 4 hours
            
        if age < limit:
            print(f"Loading from cache ({age/60:.1f}m old)...")
            try:
                return pd.read_pickle(cache_file)
            except:
                pass # Corrupt cache

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    # Save Cache
    if not data.empty:
        try:
            data.to_pickle(cache_file)
        except Exception as e:
            print(f"Failed to cache data: {e}")
    
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
