import data_loader
import features
import patterns
import model
import signals
import backtest
import visualize
from strategies.zero_dte import ZeroDTEStrategy
import pandas as pd
import argparse
import os

def run_pipeline(symbol='SPY', timeframe='15m', period='60d', mode='train', options=True, zero_dte=False):
    """
    Orchestrates the full trading bot pipeline.
    """
    print(f"--- Starting Pro Pipeline for {symbol} ({timeframe}) ---")
    
    # 1. Load Data
    df = data_loader.fetch_data(symbol, interval=timeframe, period=period)
    if df.empty: return
    
    # 2. Feature Engineering
    df = features.compute_features(df)
    
    # 3. Pattern Labeling
    df = patterns.label_data(df)
    
    # 4. Model Handling
    tm = model.TradingModel()
    if mode == 'train':
        tm.train(df)
    else:
        if not tm.load():
            print("No model found. Running train mode instead...")
            tm.train(df)
            
    # 5. Signal Generation
    signals_df = signals.generate_signals(df, tm)
    
    # 6. Apply Specialized Strategies
    if zero_dte:
        print("Applying 0DTE specialized logic...")
        zdte = ZeroDTEStrategy()
        signals_df = zdte.get_0dte_signals(signals_df)
        # Map specialized signals to the main signal column for backtester
        signals_df['signal'] = signals_df['zero_dte_signal']
    
    # 7. Backtesting
    bt = backtest.Backtester(signals_df, options_mode=options)
    equity_df = bt.run()
    
    # 8. Visualization
    journal_df = pd.read_csv('trade_journal.csv')
    visualize.plot_performance(equity_df, journal_df)
    
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Trading Bot Pro Orchestrator')
    parser.add_argument('--symbol', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--timeframe', type=str, default='15m', help='Data interval')
    parser.add_argument('--period', type=str, default='60d', help='History period')
    parser.add_argument('--mode', type=str, default='backtest', choices=['train', 'backtest'], help='Run mode')
    parser.add_argument('--options', action='store_true', default=True, help='Enable realistic options pricing')
    parser.add_argument('--zero-dte', action='store_true', help='Enable 0DTE specialized strategy')
    
    args = parser.parse_args()
    
    run_pipeline(symbol=args.symbol, timeframe=args.timeframe, period=args.period, 
                 mode=args.mode, options=args.options, zero_dte=args.zero_dte)
