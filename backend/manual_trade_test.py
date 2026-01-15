import os
import sys

# Add core to path
sys.path.append(os.path.join(os.getcwd(), 'core'))

from core.scheduler import run_auto_trade_cycle

if __name__ == "__main__":
    # Test with a small interval for quick check
    run_auto_trade_cycle(symbol='SPY', interval='15m')
