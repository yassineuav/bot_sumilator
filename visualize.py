import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_performance(equity_df, journal_df, output_dir='results'):
    """
    Generates performance visualizations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df['Datetime'], equity_df['equity'], label='Equity Curve', color='blue')
    plt.title('Account Equity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Balance ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()
    
    # 2. Drawdown Chart
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(equity_df['Datetime'], equity_df['drawdown'], color='red', alpha=0.3)
    plt.title('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'drawdown.png'))
    plt.close()
    
    # 3. Trade Distribution
    if not journal_df.empty:
        plt.figure(figsize=(10, 6))
        journal_df['pnl_pct'].hist(bins=20, color='skyblue', edgecolor='black')
        plt.title('Win/Loss Distribution')
        plt.xlabel('PnL %')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'))
        plt.close()

    print(f"Visualizations saved to {output_dir}/")

if __name__ == "__main__":
    # Test with dummy data if needed
    pass
