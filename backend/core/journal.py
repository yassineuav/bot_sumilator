import pandas as pd
import numpy as np
from datetime import datetime

class Journal:
    def __init__(self, filename='trade_journal.csv'):
        self.filename = filename
        self.trades = []
        
    def add_trade(self, trade_data):
        """
        trade_data: dict with trade details
        """
        self.trades.append(trade_data)
        
    def save(self):
        df = pd.DataFrame(self.trades)
        df.to_csv(self.filename, index=False)
        print(f"Journal saved to {self.filename}")
        return df

    def get_summary(self):
        if not self.trades:
            return "No trades recorded."
        
        df = pd.DataFrame(self.trades)
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        
        summary = {
            'total_trades': len(df),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'total_pnl_pct': df['pnl_pct'].sum(),
            'avg_pnl_pct': df['pnl_pct'].mean(),
            'max_win': df['pnl_pct'].max(),
            'max_loss': df['pnl_pct'].min(),
        }
        return summary
