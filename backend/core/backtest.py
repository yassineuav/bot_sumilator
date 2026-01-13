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
                    
                    contract_prices = self.options_engine.simulate_contract_price(
                        trade['price_history'],
                        strike_pct=1.005 if trade['type'] == 'CALL' else 0.995,
                        dte=eff_dte,
                        initial_vol=0.25,
                        steps_per_day=steps
                    )
                    curr_opt = contract_prices[-1]
                    entry_opt = contract_prices[0]
                    pnl_pct = (curr_opt - entry_opt) / entry_opt if entry_opt != 0 else 0
                else:
                    if trade['type'] == 'CALL':
                        pnl_pct = (row['Close'] - trade['entry_price']) / trade['entry_price']
                    else:
                        pnl_pct = (trade['entry_price'] - row['Close']) / trade['entry_price']
                
                trade['unrealized_pnl_val'] = trade['size'] * pnl_pct
                
                # Check Exit Conditions
                should_exit, reason, _ = self.risk_manager.check_exit_conditions(
                    trade['entry_price'], row['Close'], trade['type']
                )
                
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    should_exit, reason = True, "Stop Loss"
                elif pnl_pct >= self.risk_manager.tp_stages[0][0]:
                    should_exit, reason = True, "Take Profit"
                    
                # Opposite Signal Exit? (Optional, skipping for multi-trade logic mostly)

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
