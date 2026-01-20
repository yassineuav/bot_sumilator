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

    def get_clock(self):
        try:
            return self.api.get_clock()
        except Exception as e:
            print(f"Error getting Alpaca clock: {e}")
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

    def get_portfolio_history(self, period='1M', timeframe='1D'):
        """
        Fetches portfolio equity history.
        period: The duration of the data (e.g., '1M', '1A').
        timeframe: The resolution of the data (e.g., '1D', '15Min').
        """
        try:
            history = self.api.get_portfolio_history(period=period, timeframe=timeframe)
            return history
        except Exception as e:
            print(f"Error getting portfolio history: {e}")
            return None

    def get_closed_orders(self, limit=50):
        """
        Fetches recently closed orders.
        """
        try:
            orders = self.api.list_orders(status='closed', limit=limit, direction='desc')
            return orders
        except Exception as e:
            print(f"Error getting closed orders: {e}")
            return None

    def get_option_contract(self, symbol, expiration_date, option_type, strike_price):
        """
        Constructs an OSI Options Symbol for Alpaca/OCC standard.
        Format: SYMBOL + YYMMDD + Type(C/P) + 00000000 (Strike * 1000)
        Example: SPY240119C00475000 (SPY Jan 19 2024 475 Call)
        """
        try:
            # 1. Format Date: YYMMDD
            # expiration_date should be a datetime object or YYYY-MM-DD string
            if isinstance(expiration_date, str):
                from datetime import datetime
                dt = datetime.strptime(expiration_date, "%Y-%m-%d")
            else:
                dt = expiration_date
            
            yymmdd = dt.strftime("%y%m%d")
            
            # 2. Format Type: C or P
            type_code = 'C' if option_type.upper() in ['CALL', 'C'] else 'P'
            
            # 3. Format Strike: 00000000 (8 digits, implied 3 decimals)
            # 475.00 -> 475000 -> 00475000
            strike_int = int(float(strike_price) * 1000)
            strike_str = f"{strike_int:08d}"
            
            # Combine
            osi_symbol = f"{symbol}{yymmdd}{type_code}{strike_str}"
            return osi_symbol
            
        except Exception as e:
            print(f"Error constructing option symbol: {e}")
            return None
