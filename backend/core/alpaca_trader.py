import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

class AlpacaTrader:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )

    def get_account(self):
        try:
            return self.api.get_account()
        except Exception as e:
            print(f"Error getting Alpaca account: {e}")
            return None

    def place_order(self, symbol, qty, side, type='market', time_in_force='gtc'):
        """
        side: 'buy' or 'sell'
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force
            )
            print(f"Order placed: {order.id} for {qty} shares of {symbol}")
            return order
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    def get_positions(self):
        try:
            return self.api.list_positions()
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def close_position(self, symbol):
        try:
            self.api.close_position(symbol)
            print(f"Closed position for {symbol}")
            return True
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
            return False
