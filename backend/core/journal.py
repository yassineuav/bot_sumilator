import pandas as pd
import numpy as np
import os
from datetime import datetime
JOURNAL_DIR = 'trade_journals'
DEFAULT_JOURNAL_NAME = 'trade_journal.csv'

class Journal:
    def __init__(self, symbol=None, interval=None, filename=None, project_name='default'):
        self.symbol = symbol
        self.interval = interval
        self.filename = filename
        self.project_name = project_name
        self.trades = []
        
    def get_journal_path(self):
        if self.filename:
            return self.filename
            
        base_dir = os.path.join(JOURNAL_DIR, self.project_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        if self.symbol and self.interval:
            return os.path.join(base_dir, f"journal_{self.symbol}_{self.interval}.csv")
        return DEFAULT_JOURNAL_NAME

    def add_trade(self, trade_data):
        """
        trade_data: dict with trade details
        """
        self.trades.append(trade_data)
        
    def save(self):
        df = pd.DataFrame(self.trades)
        path = self.get_journal_path()
        df.to_csv(path, index=False)
        print(f"Journal saved to {path}")
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
