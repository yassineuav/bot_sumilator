import pandas as pd
import numpy as np
from risk import RiskManager
from journal import Journal
from options_engine import OptionsEngine


class Backtester:
    def __init__(self, signals_df, symbol='SPY', initial_balance=1000.0, options_mode=True, interval='15m', target_dte=0.0,
                 risk_per_trade_pct=0.20, stop_loss_pct=0.10, take_profit_pct=0.50, project_name='default'):
        self.df = signals_df
        self.symbol = symbol
        self.risk_manager = RiskManager(
            starting_balance=initial_balance,
            risk_per_trade_pct=risk_per_trade_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        self.journal = Journal(symbol=symbol, interval=interval, project_name=project_name)
        self.options_engine = OptionsEngine()
        self.balance = initial_balance
        self.equity_curve = []
        self.options_mode = options_mode
        self.interval = interval
        self.target_dte = target_dte if target_dte is not None else 0.0
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

                    # STRIKE: Use stored strike
                    strike = trade.get('strike')
                    if strike is None:
                         # Fallback for old trades (shouldn't happen in new runs)
                         offset = 2.5
                         if trade['type'] == 'CALL':
                              strike = round(trade['entry_price'] - offset)
                         else:
                              strike = round(trade['entry_price'] + offset)
                    
                    strike_pct = strike / trade['entry_price']

                    
                    # OPTIMIZATION: Calculate only current option price
                    duration_steps = len(trade['price_history']) - 1
                    current_dte = eff_dte - (duration_steps / float(steps))
                    T = max(0, current_dte / 365.0)
                    
                    # Option Type
                    # If strike was derived from strike_pct logic earlier, we need to know call/put
                    # trade['type'] has 'CALL'/'PUT'
                    opt_type = 'call' if trade['type'] == 'CALL' else 'put'
                    
                    # Current Option Price
                    res = self.options_engine.black_scholes(
                        S=row['Close'],
                        K=strike,
                        T=T,
                        r=0.05,
                        sigma=dynamic_vol,
                        option_type=opt_type
                    )
                    curr_opt = res['price']
                    
                    # Entry Option Price
                    # Should be stored in trade! If not (legacy), calculate it once
                    if 'entry_opt_price' not in trade:
                        # Back-calculate entry price (T=eff_dte, S=entry_price)
                        res_entry = self.options_engine.black_scholes(
                            S=trade['entry_price'],
                            K=strike,
                            T=eff_dte/365.0,
                            r=0.05,
                            sigma=initial_vol_at_entry if 'initial_vol' in trade else dynamic_vol, # Simplified
                            option_type=opt_type
                        )
                        trade['entry_opt_price'] = max(0.10, res_entry['price'])
                        
                    entry_opt = trade['entry_opt_price']
                    
                    pnl_pct = (curr_opt - entry_opt) / entry_opt
                else:
                    if trade['type'] == 'CALL':
                        pnl_pct = (row['Close'] - trade['entry_price']) / trade['entry_price']
                    else:
                        pnl_pct = (trade['entry_price'] - row['Close']) / trade['entry_price']
                
                trade['unrealized_pnl_val'] = trade['size'] * pnl_pct
                
                # Check Exit Conditions
                # Use RiskManager for Stop Loss (pct) and Take Profit (pct)
                # We calculate PnL manually for options above, so we skip check_exit_conditions PnL calc.
                
                # Wait, RiskManager.check_exit_conditions(entry, current, type) calculates PnL based on Underlying logic!
                # If we are in Options Mode, pnl_pct is Option PnL. 
                # We should probably bypass check_exit_conditions PnL calc and just check the thresholds logic.
                
                # Implementing fix inline here to avoid changing RiskManager signature too much or confusing it.
                should_exit = False
                reason = ""
                
                if pnl_pct <= -self.risk_manager.stop_loss_pct:
                    should_exit, reason = True, "Stop Loss"
                elif pnl_pct >= self.risk_manager.take_profit_pct:
                    should_exit, reason = True, f"Take Profit ({self.risk_manager.take_profit_pct*100:.0f}%)"
                    
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
                        'confidence': trade['confidence'],
                        'strike': trade.get('strike')
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
                        
                    # STRIKE SELECTION (ITM 0DTE)
                    # User Request: Strike $1 to $4 ITM (In The Money)
                    # For CALL: Strike < Price (Price - 2.5)
                    # For PUT: Strike > Price (Price + 2.5)
                    offset = 2.5 # Midpoint of $1-$4
                    
                    if row['signal'] == 'CALL':
                        # Strike below price (ITM)
                        raw_strike = row['Close'] - offset
                        strike = round(raw_strike) # Round to nearest integer strike
                    else:
                        # Strike above price (ITM)
                        raw_strike = row['Close'] + offset
                        strike = round(raw_strike)

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
                    
                    
                    # PRICE FILTER REMOVED for ITM Strategy
                    # ITM options will be expensive (Intrinsic > $2.5), so we cannot use the cheap option filter.
                    # if not (0.10 <= entry_option_price <= 0.30):
                    #     check_price = False
                    
                    if entry_option_price <= 0.05:
                        check_price = False # Still filter out garbage/untradeable

                    size = self.risk_manager.calculate_position_size(self.balance)
                    if size > 10:
                        new_trade = {
                            'entry_time': row['Datetime'],
                            'type': row['signal'],
                            'entry_price': row['Close'],
                            'size': size,
                            'strike': strike, # STORE STRIKE
                            'confidence': row['confidence'],
                            'unrealized_pnl_val': 0,
                            'entry_opt_price': max(0.10, entry_option_price), # STORE ENTRY OPT PRICE
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


    # ... (rest of class)
        
    # (run method remains same)

    # ... (code for run method should be preserved, skipping for brevity in this tool call if possible, but replace_file_content requires context. 
    # Actually, modify calculate_current_options_pnl which is at the bottom)

    def calculate_current_options_pnl(self, active_trade):
        """
        Helper to get the current option PnL.
        """
        # Reproduce Strike Selection (ITM $2.5)
        # We need the ENTRY PRICE of the stock to reconstruct the strike?
        # active_trade['entry_price'] stores the Stock Price at entry.
        if 'strike' in active_trade:
             strike = active_trade['strike']
             strike_pct = strike / active_trade['entry_price']
        else:
             # Fallback
             offset = 2.5
             if active_trade['type'] == 'CALL':
                  strike = round(active_trade['entry_price'] - offset)
             else:
                  strike = round(active_trade['entry_price'] + offset)
             strike_pct = strike / active_trade['entry_price']
             
        # NOTE: limit_strike_pct logic in options_engine rely on pct usually? 
        # But simulate_contract_price takes strike_pct.
        # Let's use the explicit calculated pct.
        
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
