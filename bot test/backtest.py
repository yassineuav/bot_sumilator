import pandas as pd
import numpy as np
from risk import RiskManager
from journal import Journal
from options_engine import OptionsEngine

class Backtester:
    def __init__(self, signals_df, initial_balance=1000.0, options_mode=True):
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
        
        active_trade = None
        
        for i, row in self.df.iterrows():
            # Apply balance tracking
            self.equity_curve.append({
                'Datetime': row['Datetime'],
                'equity': self.balance + (active_trade['unrealized_pnl_val'] if active_trade else 0)
            })
            
            # 1. If we have an active trade, check exit conditions
            if active_trade:
                active_trade['price_history'].append(row['Close'])
                
                # Use OptionsEngine if enabled
                if self.options_mode:
                    # Calculate current pnl using options pricing
                    pnl_pct = self.calculate_current_options_pnl(active_trade)
                else:
                    if active_trade['type'] == 'CALL':
                        pnl_pct = (row['Close'] - active_trade['entry_price']) / active_trade['entry_price']
                    else:
                        pnl_pct = (active_trade['entry_price'] - row['Close']) / active_trade['entry_price']

                # Check SL/TP from Risk Manager
                should_exit, reason, _ = self.risk_manager.check_exit_conditions(
                    active_trade['entry_price'], 
                    row['Close'], 
                    active_trade['type']
                )
                
                # Override reason if options-based PnL hits SL/TP levels
                # (RiskManager expects underlying % usually, but we can adapt)
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    should_exit, reason = True, "Options Stop Loss"
                elif pnl_pct >= self.risk_manager.tp_stages[0][0]:
                    should_exit, reason = True, "Options Take Profit"

                # Check for opposite signal as exit
                if not should_exit:
                    if active_trade['type'] == 'CALL' and row['signal'] == 'PUT':
                        should_exit, reason = True, "Opposite Signal"
                    elif active_trade['type'] == 'PUT' and row['signal'] == 'CALL':
                        should_exit, reason = True, "Opposite Signal"

                if should_exit:
                    exit_val = active_trade['size'] * (1 + pnl_pct)
                    pnl_val = exit_val - active_trade['size']
                    self.balance += pnl_val
                    
                    # Log trade
                    trade_record = {
                        'entry_time': active_trade['entry_time'],
                        'exit_time': row['Datetime'],
                        'type': active_trade['type'],
                        'entry_price': active_trade['entry_price'],
                        'exit_price': row['Close'],
                        'pnl_pct': pnl_pct,
                        'pnl_val': pnl_val,
                        'exit_reason': reason,
                        'confidence': active_trade['confidence']
                    }
                    self.journal.add_trade(trade_record)
                    active_trade = None
                    continue

            # 2. If no active trade, check for entry signals
            if not active_trade and row['signal'] != 'NO TRADE':
                size = self.risk_manager.calculate_position_size(self.balance)
                if size > 10: # Minimum trade size
                    active_trade = {
                        'entry_time': row['Datetime'],
                        'type': row['signal'],
                        'entry_price': row['Close'],
                        'size': size,
                        'confidence': row['confidence'],
                        'unrealized_pnl_val': 0,
                        'price_history': [row['Close']]
                    }
                    
        # Finalize
        self.journal.save()
        summary = self.journal.get_summary()
        print("Backtest Complete.")
        print(summary)
        return pd.DataFrame(self.equity_curve)

    def calculate_current_options_pnl(self, active_trade):
        """
        Helper to get the current option PnL.
        """
        strike_pct = 1.005 if active_trade['type'] == 'CALL' else 0.995
        # 0.5 DTE for realistic intraday movement
        contract_prices = self.options_engine.simulate_contract_price(
            active_trade['price_history'],
            strike_pct=strike_pct,
            dte=0.5,
            initial_vol=0.25
        )
        entry_opt = contract_prices[0]
        curr_opt = contract_prices[-1]
        if entry_opt == 0: return 0
        return (curr_opt - entry_opt) / entry_opt

if __name__ == "__main__":
    pass
