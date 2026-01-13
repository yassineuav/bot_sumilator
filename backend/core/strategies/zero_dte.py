import pandas as pd
import numpy as np
from options_engine import OptionsEngine

class ZeroDTEStrategy:
    def __init__(self):
        self.options_engine = OptionsEngine()
        
    def get_0dte_signals(self, df):
        """
        Specialized 1m-level signals for 0DTE.
        Looking for momentum spikes in the first 30 mins or last 30 mins.
        """
        df = df.copy()
        df['time'] = df['Datetime'].dt.time
        
        # Identify high volatility windows
        market_open = pd.Timestamp("09:30").time()
        morning_cutoff = pd.Timestamp("10:30").time()
        afternoon_start = pd.Timestamp("15:00").time()
        market_close = pd.Timestamp("16:00").time()
        
        df['is_0dte_window'] = ((df['time'] >= market_open) & (df['time'] <= morning_cutoff)) | \
                               ((df['time'] >= afternoon_start) & (df['time'] <= market_close))
                               
        # Using existing signal logic but filtering for these windows
        # and requiring higher confidence/momentum.
        df['zero_dte_signal'] = 'NO TRADE'
        mask_call = (df['is_0dte_window']) & (df['signal'] == 'CALL') & (df['confidence'] > 0.7)
        mask_put = (df['is_0dte_window']) & (df['signal'] == 'PUT') & (df['confidence'] > 0.7)
        
        df.loc[mask_call, 'zero_dte_signal'] = 'CALL'
        df.loc[mask_put, 'zero_dte_signal'] = 'PUT'
        
        return df

    def calculate_option_pnl(self, entry_price, price_history, type='CALL'):
        """
        Calculates PnL using the OptionsEngine for a 0DTE contract.
        """
        # Assume 0.5% OTM strike for better leverage
        strike_pct = 1.005 if type == 'CALL' else 0.995
        # 0 DTE (1 day or less)
        contract_prices = self.options_engine.simulate_contract_price(
            price_history, 
            strike_pct=strike_pct, 
            dte=0.1, # 1/10th of a day for 0DTE feel
            initial_vol=0.25
        )
        
        entry_opt_price = contract_prices[0]
        exit_opt_price = contract_prices[-1]
        
        if entry_opt_price == 0: return 0
        pnl_pct = (exit_opt_price - entry_opt_price) / entry_opt_price
        return pnl_pct
