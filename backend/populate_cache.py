
import os
import sys
import django

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trading_backend.settings')
django.setup()

from core.data_loader import fetch_data

def populate():
    symbol = 'SPY'
    print(f"Populating cache for {symbol}...")
    
    # Fetch 15m 60d
    print("Fetching 15m 60d...")
    fetch_data(symbol, interval='15m', period='60d')
    
    # Fetch 1d 10y
    print("Fetching 1d 10y...")
    fetch_data(symbol, interval='1d', period='10y')
    
    # Fetch 1h 2y (730d)
    print("Fetching 1h 730d...")
    fetch_data(symbol, interval='1h', period='730d')
    
    print("Cache population complete.")

if __name__ == "__main__":
    populate()
