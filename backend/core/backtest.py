import pandas as pd
import numpy as np
from risk import RiskManager
from journal import Journal
from options_engine import OptionsEngine

class Backtester:
    def __init__(self, signals_df, initial_balance=1000.0, options_mode=True, interval='15m', target_dte=0.0):
        self.df = signals_df
        self.risk_manager = RiskManager(starting_balance=initial_balance)
        self.journal = Journal()
        self.options_engine = OptionsEngine()
        self.balance = initial_balance
        self.equity_curve = []
        self.options_mode = options_mode
        self.price_history_buffer = [] # To track spot prices for options simulation
        
    def run(self):
        print(f"Starting backtest (Options: {self.options_mode}) with ${self.balance}...")
        
        active_trades = []
        trades_today = 0
        current_day = None
        
        # Determine steps per day for options pricing
        steps_map = {
            '1m': 390, '2m': 195, '5m': 78, '15m': 26, 
            '30m': 13, '1h': 7, '90m': 5, '4h': 2, '1d': 1
        }
        steps = steps_map.get(self.interval, 26)
        
        for i, row in self.df.iterrows():
            # Check for new day to reset daily counter
            row_date = row['Datetime'].date()
            if current_day != row_date:
                current_day = row_date
                trades_today = 0
                
            # Apply balance tracking (equity = balance + unrealized pnl of all trades)
            unrealized_total = sum(t['unrealized_pnl_val'] for t in active_trades)
            self.equity_curve.append({
                'Datetime': row['Datetime'],
                'equity': self.balance + unrealized_total
            })
            
            # 1. Manage Active Trades (Check Exits)
            # Create a copy to modify list while iterating
            for trade in active_trades[:]:
                trade['price_history'].append(row['Close'])
                
                # Calculate PnL
                if self.options_mode:
                    # Use effective DTE. If 0DTE, we just use the remainder of the day?
                    # or fixed DTE provided? 
                    eff_dte = max(0.4, self.target_dte)
                    
                    # Calculate dynamic volatility from recent price history (e.g. last 20 steps)
                    # If history too short, default to 0.20
                    if len(trade['price_history']) > 2:
                        hist_series = pd.Series(trade['price_history'])
                        returns = hist_series.pct_change().dropna()
                        # Annualize
                        annual_factor = (252 * steps) ** 0.5
                        if len(returns) > 1:
                            dynamic_vol = returns.std() * annual_factor
                            if pd.isna(dynamic_vol) or dynamic_vol < 0.15:
                                dynamic_vol = 0.15
                        else:
                            dynamic_vol = 0.15
                    else:
                        dynamic_vol = 0.15

                    contract_prices = self.options_engine.simulate_contract_price(
                        trade['price_history'],
                        strike_pct=1.005 if trade['type'] == 'CALL' else 0.995,
                        dte=eff_dte,
                        initial_vol=dynamic_vol,
                        steps_per_day=steps
                    )
                    
                    # Clamp Minimum Option Price to $0.10 to prevent unrealistic % gains from penny options
                    entry_opt = max(0.10, contract_prices[0])
                    # We must also clamp the current price if we want consistent PnL logic, 
                    # but actually we just care that the ENTRY wasn't unrealistically cheap.
                    # The exit price depends on the market.
                    curr_opt = contract_prices[-1]
                    
                    pnl_pct = (curr_opt - entry_opt) / entry_opt
                else:
                    if trade['type'] == 'CALL':
                        pnl_pct = (row['Close'] - trade['entry_price']) / trade['entry_price']
                    else:
                        pnl_pct = (trade['entry_price'] - row['Close']) / trade['entry_price']
                
                trade['unrealized_pnl_val'] = trade['size'] * pnl_pct
                
                # Check Exit Conditions
                # DEFAULT EXIT: Use RiskManager for Stop Loss (pct)
                # PROFIT TARGET: User wants to sell at $0.40 or more
                should_exit = False
                reason = "Target"
                
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    should_exit, reason = True, "Stop Loss"
                elif self.options_mode and curr_opt >= 0.40:
                    should_exit, reason = True, "Price Target ($0.40)"
                elif pnl_pct >= self.risk_manager.tp_stages[0][0]:
                     # Fallback to percentage TP if reached before price target
                    should_exit, reason = True, "Take Profit (%)"
                    
                if should_exit:
                    exit_val = trade['size'] * (1 + pnl_pct)
                    pnl_val = exit_val - trade['size']
                    self.balance += pnl_val
                    
                    trade_record = {
                        'entry_time': trade['entry_time'],
                        'exit_time': row['Datetime'],
                        'type': trade['type'],
                        'entry_price': trade['entry_price'],
                        'exit_price': row['Close'],
                        'pnl_pct': pnl_pct,
                        'pnl_val': pnl_val,
                        'exit_reason': reason,
                        'confidence': trade['confidence']
                    }
                    self.journal.add_trade(trade_record)
                    active_trades.remove(trade)

            # 2. Check Entries
            # Enforce limits: Max 5 positions at once, Max 5 trades per day
            if (len(active_trades) < 5 and 
                trades_today < 5 and 
                row['signal'] != 'NO TRADE'):
                
                # OPTION PRICE FILTER: Only buy if Option Price is between $0.10 and $0.30
                check_price = True
                entry_option_price = 0.0
                
                if self.options_mode:
                    # needed to estimate price
                    lookback = 20
                    past_data = self.df.iloc[max(0, i-lookback):i+1]
                    if len(past_data) > 2:
                        returns = past_data['Close'].pct_change().dropna()
                        annual_factor = (252 * steps) ** 0.5
                        vol = returns.std() * annual_factor
                        if pd.isna(vol) or vol < 0.15: vol = 0.15
                    else:
                        vol = 0.25
                        
                    # STRIKE: Out of money strik price option by 2 or 3$ far.
                    # For a $600 stock, $3 is 0.5%. Using 0.005 offset.
                    strike_pct = 1.005 if row['signal'] == 'CALL' else 0.995
                    strike = row['Close'] * strike_pct
                    dte_days = max(0.4, self.target_dte)
                    T = dte_days / 365.0
                    
                    res = self.options_engine.black_scholes(
                        S=row['Close'], 
                        K=strike, 
                        T=T, 
                        r=0.05, 
                        sigma=vol, 
                        option_type='call' if row['signal'] == 'CALL' else 'put'
                    )
                    entry_option_price = res['price']
                    
                    # THE FILTER ($0.1 to $0.3)
                    if not (0.10 <= entry_option_price <= 0.30):
                        check_price = False
                
                if check_price:
                    size = self.risk_manager.calculate_position_size(self.balance)
                    if size > 10:
                        new_trade = {
                            'entry_time': row['Datetime'],
                            'type': row['signal'],
                            'entry_price': row['Close'],
                            'size': size,
                            'confidence': row['confidence'],
                            'unrealized_pnl_val': 0,
                            'price_history': [row['Close']]
                        }
                        active_trades.append(new_trade)
                        trades_today += 1
                    
        # Finalize
        self.journal.save()
        summary = self.journal.get_summary()
        print("Backtest Complete.")
        print(summary)
        return pd.DataFrame(self.equity_curve)

    def __init__(self, signals_df, initial_balance=1000.0, options_mode=True, interval='15m', target_dte=0.0):
        self.df = signals_df
        self.risk_manager = RiskManager(starting_balance=initial_balance)
        self.journal = Journal()
        self.options_engine = OptionsEngine()
        self.balance = initial_balance
        self.equity_curve = []
        self.options_mode = options_mode
        self.interval = interval
        self.target_dte = target_dte if target_dte is not None else 0.0
        self.price_history_buffer = [] 
        
    # (run method remains same)

    # ... (code for run method should be preserved, skipping for brevity in this tool call if possible, but replace_file_content requires context. 
    # Actually, modify calculate_current_options_pnl which is at the bottom)

    def calculate_current_options_pnl(self, active_trade):
        """
        Helper to get the current option PnL.
        """
        strike_pct = 1.005 if active_trade['type'] == 'CALL' else 0.995
        
        # Determine steps per day based on interval
        steps_map = {
            '1m': 390,
            '2m': 195,
            '5m': 78,
            '15m': 26,
            '30m': 13,
            '1h': 7,
            '90m': 5,
            '4h': 2,
            '1d': 1
        }
        steps = steps_map.get(self.interval, 26)
        
        # Use configured DTE. For 0DTE, start with 0.5 (half day remaining) effectively?
        # Actually if DTE=0, black scholes might return intrinsic immediately.
        # Let's use user supplied DTE but ensure it's at least enough for the trade duration simulation.
        # If target_dte is 0 (0DTE), we simulate starting morning of expiry -> 1 day (or 0.4 day).
        eff_dte = max(0.4, self.target_dte)
        
        contract_prices = self.options_engine.simulate_contract_price(
            active_trade['price_history'],
            strike_pct=strike_pct,
            dte=eff_dte,
            initial_vol=0.25,
            steps_per_day=steps
        )
        entry_opt = contract_prices[0]
        curr_opt = contract_prices[-1]
        
        # Handle dict return from options_engine (if updated) or float (if old, but we updated it)
        # Wait, we updated options_engine to return list of floats usually, 
        # but at T=0 it returns dict? 
        # Ah, in emulate loop: contract_prices.append(res['price']). 
        # So contract_prices is always list of floats. Good.
        
        if entry_opt == 0: return 0
        return (curr_opt - entry_opt) / entry_opt

if __name__ == "__main__":
    pass
