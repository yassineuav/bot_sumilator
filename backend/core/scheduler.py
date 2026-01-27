from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import os
import sys
import pandas as pd
import numpy as np
import traceback

# Ensure we can import core modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

import data_loader
import features
import model
import signals
from core.alpaca_trader import AlpacaTrader

import feature_pipeline

def run_auto_trade_cycle(symbol='SPY', interval='15m'):
    trader = AlpacaTrader()
    clock = trader.get_clock()
    if not clock or not clock.is_open:
        # Silent skip or minimal log if needed, but user requested 'only when market open'
        # We will skip printing "Starting..." to avoid spam when closed.
        return

    print(f"[{datetime.datetime.now()}] Starting auto-trade cycle for {symbol} ({interval})")
    
    try:
        
        # 1. Load Model
        tm = model.TradingModel(symbol=symbol, interval=interval)
        if not tm.load():
            print(f"Model not found for {symbol} {interval}. Skipping cycle.")
            return

        # 2. Fetch Data
        # We need enough data for features
        # Use MultiTimeframePipeline to ensure we have 1d, 1h, and base features
        pipeline = feature_pipeline.MultiTimeframePipeline(symbol)
        try:
            df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period='60d')
        except Exception as e:
            print(f"Error fetching/preparing data: {e}")
            return

        if df.empty:
            print("No data found (empty dataframe). Skipping.")
            return
            
        # 3. Predict on most recent candle
        current_data = df.iloc[[-1]].copy()
        probs = tm.predict_proba(current_data)[0]
        pred_idx = np.argmax(probs)
        signal_map = {0: 'PUT', 1: 'NEUTRAL', 2: 'CALL'}
        signal = signal_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        print(f"Signal: {signal} (Confidence: {confidence:.2%})")
        
        if signal == 'NEUTRAL':
            print("Neutral signal. No action.")
            return
            
        # 4. Execute Trade on Alpaca
        # Check current positions
        positions = trader.get_positions()
        has_position = any(p.symbol == symbol for p in positions)
        
        if has_position:
            print(f"Already have a position in {symbol}. Skipping entry.")
            # Future logic: check for exit signals or TP/SL
            return
            
        # Basic position sizing: 1% of equity or fixed shares
        account = trader.get_account()
        if not account:
            return
            
        equity = float(account.equity)
        msg = f"Account Equity: ${equity:.2f}"
        print(msg)
        
        # For paper trading, let's start with a small fixed amount or 10% of equity
        # Since this is a stock trading bot (Alpaca basic), we trade shares.
        # If the user wants options, the logic would be much more complex.
        # For now, we'll trade the underlying symbol.
        
        price = float(current_data['Close'].iloc[0])
        qty = int((equity * 0.1) / price)
        if qty <= 0: qty = 1
        
        side = 'buy' if signal == 'CALL' else 'sell' # Simplistic: Buy for Call, Sell for Put (Short)

        print(f"Placing {side} order for {qty} shares of {symbol}...")
        trader.place_order(symbol, qty, side)
        
    except Exception as e:
        print(f"Error in auto-trade cycle: {e}")
        traceback.print_exc()

def check_system_health():
    try:
        trader = AlpacaTrader()
        clock = trader.get_clock()
        
        # Only log heartbeat if market is open, as requested
        if clock and clock.is_open:
            print(f"[{datetime.datetime.now()}] Heartbeat: Backend health check...")
            # System status log moved to on-demand (views.py) per user request
            # account = trader.get_account()
            # if account:
            #     print(f"[{datetime.datetime.now()}] System Status: ONLINE (Alpaca Connected)")
            # else:
            #     print(f"[{datetime.datetime.now()}] System Status: WARNING (Alpaca Disconnected)")
        # Else: Market closed, stay silent
    except Exception as e:
        # Only log errors if we really need to, keeping silent for now to satisfy request
        pass

def start_scheduler():
    scheduler = BackgroundScheduler()
    # Adding jobs for both 5 and 15 minutes as requested
    scheduler.add_job(run_auto_trade_cycle, 'interval', minutes=15, args=['SPY', '15m'], id='trade_spy_15m')
    scheduler.add_job(run_auto_trade_cycle, 'interval', minutes=5, args=['SPY', '5m'], id='trade_spy_5m')
    scheduler.add_job(check_system_health, 'interval', minutes=1, id='system_health_check')
    
    scheduler.start()
    print("Background Scheduler Started.")
    return scheduler
