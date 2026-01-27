class RiskManager:
    def __init__(self, starting_balance=1000.0, risk_per_trade_pct=0.20, stop_loss_pct=0.10, take_profit_pct=0.50):
        self.balance = starting_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Simple TP stages for now, using the single TP target
        self.tp_stages = [
            (take_profit_pct, 1.0)
        ]
        self.max_open_trades = 5
        
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
            
        # Check TP
        if pnl_pct >= self.take_profit_pct:
            return True, f"Take Profit ({self.take_profit_pct*100:.0f}%)", pnl_pct
            
        return False, None, pnl_pct

if __name__ == "__main__":
    rm = RiskManager()
    size = rm.calculate_position_size(1000)
    print(f"Position Size: ${size}")
    
    exit_sl, reason_sl, pnl_sl = rm.check_exit_conditions(100, 89, 'CALL')
    print(f"Exit (SL): {exit_sl}, Reason: {reason_sl}, PnL: {pnl_sl}")
    
    exit_tp, reason_tp, pnl_tp = rm.check_exit_conditions(100, 201, 'CALL')
    print(f"Exit (TP): {exit_tp}, Reason: {reason_tp}, PnL: {pnl_tp}")
