class RiskManager:
    def __init__(self, starting_balance=1000.0, risk_per_trade_pct=0.20):
        self.balance = starting_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = 0.10 # 10%
        self.tp_stages = [
            (1.0, 0.25), # 100% TP for 25% of position
            (2.0, 0.25), # 200% TP for 25% of position
            (3.0, 0.25), # 300% TP for 25% of position
            (5.0, 0.25)  # 500% TP for 25% of position
        ]
        self.max_open_trades = 1
        
    def calculate_position_size(self, current_balance):
        return current_balance * self.risk_per_trade_pct

    def check_exit_conditions(self, entry_price, current_price, trade_type):
        """
        Returns (should_exit, reason, pnl_pct)
        """
        if trade_type == 'CALL':
            pnl_pct = (current_price - entry_price) / entry_price
        else: # PUT
            pnl_pct = (entry_price - current_price) / entry_price
            
        # Simplistic exit for backtesting: Check SL
        if pnl_pct <= -self.stop_loss_pct:
            return True, "Stop Loss", -self.stop_loss_pct
            
        # Check TP stages (simple version returns True if first TP hit for backtest simplicity, 
        # or we could simulate partials in backtest.py)
        if pnl_pct >= self.tp_stages[0][0]:
            return True, "Take Profit", pnl_pct
            
        return False, None, pnl_pct

if __name__ == "__main__":
    rm = RiskManager()
    size = rm.calculate_position_size(1000)
    print(f"Position Size: ${size}")
    
    exit_sl, reason_sl, pnl_sl = rm.check_exit_conditions(100, 89, 'CALL')
    print(f"Exit (SL): {exit_sl}, Reason: {reason_sl}, PnL: {pnl_sl}")
    
    exit_tp, reason_tp, pnl_tp = rm.check_exit_conditions(100, 201, 'CALL')
    print(f"Exit (TP): {exit_tp}, Reason: {reason_tp}, PnL: {pnl_tp}")
