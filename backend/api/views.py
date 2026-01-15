from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Trade, Performance
from .serializers import TradeSerializer, PerformanceSerializer
import sys
import os
import pandas as pd
import numpy as np
import traceback

# Add core to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

import data_loader
import features
import patterns
import model
import signals
import backtest
from strategies.zero_dte import ZeroDTEStrategy
from risk import RiskManager

@api_view(['GET'])
def get_dashboard_data(request):
    trades = Trade.objects.all().order_by('-entry_time')[:10]
    perf = Performance.objects.all()
    
    return Response({
        'latest_trades': TradeSerializer(trades, many=True).data,
        'performance': PerformanceSerializer(perf, many=True).data
    })

@api_view(['POST'])
def train_model(request):
    symbol = request.data.get('symbol', 'SPY')
    intervals = request.data.get('intervals', ['5m', '15m', '30m', '1h'])
    
    results = {}
    
    for interval in intervals:
        try:
            # Determine period dynamically
            if interval in ['1m', '2m']:
                period = '7d'
            elif interval in ['5m', '15m', '30m']:
                period = '60d'
            elif interval in ['1h', '90m']:
                period = '730d' # 2 years
            else: # 4h, 1d
                period = 'max'

            # Fetch and process data
            df = data_loader.fetch_data(symbol, interval=interval, period=period)
            if df.empty:
                results[interval] = "No data"
                continue
                
            df = features.compute_features(df)
            df = patterns.label_data(df)
            
            # Train model
            tm = model.TradingModel(symbol=symbol, interval=interval)
            tm.train(df)
            results[interval] = "Success"
        except Exception as e:
            results[interval] = str(e)
            
    return Response({'status': 'Training complete', 'details': results})

@api_view(['POST'])
def run_backtest(request):
    # Parameters
    symbol = request.data.get('symbol', 'SPY')
    interval = request.data.get('interval', '1h')
    
    # Debug: Check raw inputs
    print(f"DEBUG: Raw request data: {request.data}")
    
    initial_balance = float(request.data.get('initial_balance', 1000))
    risk_pct = float(request.data.get('risk_pct', 20)) / 100.0
    stop_loss = float(request.data.get('stop_loss', 10)) / 100.0
    take_profit = float(request.data.get('take_profit', 50)) / 100.0
    zero_dte = request.data.get('zero_dte', False)
    
    # Determine max period based on interval limits
    if interval in ['1m', '2m']:
        period = '7d'
    elif interval in ['5m', '15m', '30m', '90m']:
       period = '60d'
    else:
       period = '1y' 
    
    try:
        df = data_loader.fetch_data(symbol, interval=interval, period=period)
        if df.empty:
            return Response({'error': 'No data found'}, status=400)
            
        df = features.compute_features(df)
        df = patterns.label_data(df)
        
        tm = model.TradingModel(symbol=symbol, interval=interval)
        if not tm.load():
            tm.train(df)
            
        signals_df = signals.generate_signals(df, tm)
        
        if zero_dte:
            zdte = ZeroDTEStrategy()
            signals_df = zdte.get_0dte_signals(signals_df)
            signals_df['signal'] = signals_df['zero_dte_signal']
            
        # Target DTE: 0 for 0DTE (intraday), 30 for Swing (if not 0DTE)
        target_dte = 0.0 if zero_dte else 30.0
            
        bt = backtest.Backtester(
            signals_df, 
            symbol=symbol,
            initial_balance=initial_balance, 
            interval=interval,
            target_dte=target_dte
        )
        # Update risk parameters
        bt.risk_manager.risk_per_trade_pct = risk_pct
        bt.risk_manager.stop_loss_pct = stop_loss
        
        # DEBUG PRINTS
        print(f"DEBUG: take_profit type: {type(take_profit)}, value: {take_profit}")
        tp_s = [(take_profit, 1.0)] # List of Tuple
        print(f"DEBUG: tp_stages structure: {tp_s}")
        bt.risk_manager.tp_stages = tp_s
        
        equity_df = bt.run()
        
        # Serialize equity curve
        equity_data = equity_df[['Datetime', 'equity']].copy()
        equity_data['Datetime'] = equity_data['Datetime'].astype(str)
        
        return Response({
            'equity_curve': equity_data.to_dict(orient='records'),
            'final_balance': bt.balance,
            'total_trades': len(bt.journal.trades),
            'summary': bt.journal.get_summary(),
            'trades': bt.journal.trades
        })
    except Exception as e:
        print("ERROR IN BACKTEST:")
        traceback.print_exc()
        return Response({'error': str(e), 'traceback': traceback.format_exc()}, status=500)

@api_view(['POST'])
def sync_trades(request):
    try:
        from core.journal import Journal
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '15m')
        
        j = Journal(symbol=symbol, interval=interval)
        csv_path = j.get_journal_path()
        
        if not os.path.exists(csv_path):
            return Response({'error': f'No trade details found for {symbol} {interval} ({csv_path} missing)'}, status=404)
            
        df = pd.read_csv(csv_path)
        count = 0
        
        Trade.objects.all().delete()
        
        for _, row in df.iterrows():
            Trade.objects.create(
                symbol='SPY', 
                type=row.get('type', 'CALL'),
                entry_time=pd.to_datetime(row['entry_time']),
                entry_price=row.get('entry_price', 0),
                exit_time=pd.to_datetime(row['exit_time']),
                exit_price=row.get('exit_price', 0),
                pnl_val=row.get('pnl_val', 0),
                pnl_pct=row.get('pnl_pct', 0),
                confidence=row.get('confidence', 0),
                reason=row.get('exit_reason', '')
            )
            count += 1
            
        return Response({'status': f'Synced {count} trades'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def get_prediction(request):
    try:
        from core.options_engine import OptionsEngine
        
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '15m')
        stop_loss_pct = float(request.data.get('stop_loss', 10)) / 100.0
        take_profit_pct = float(request.data.get('take_profit', 50)) / 100.0
        
        # 1. Load Model
        tm = model.TradingModel(symbol=symbol, interval=interval)
        if not tm.load():
            return Response({'error': f'Model not found for {symbol} {interval}. Please train first.'}, status=400)
            
        # 2. Fetch Data (Need enough for features)
        df = data_loader.fetch_data(symbol, interval=interval, period='60d') # Need context for indicators
        if df.empty:
            return Response({'error': 'No data found'}, status=500)
            
        # Compute Volatility (Annualized Std Dev of Returns)
        # We need this before computing features might drop NaN, but features also computes returns
        # Let's compute simple volatility here for Black Scholes
        df['returns'] = df['Close'].pct_change()
        # Use last 20 periods for volatility
        params_vol = df['returns'].tail(20).std() * (252**0.5) 
        # Note: 252 is for daily. If interval is intraday, we need to scale differently?
        # Actually Black Scholes expects Annualized Vol.
        # If returns are 15m returns:
        # Annualized Vol = std(15m) * sqrt(periods_per_year)
        # periods_per_day ~ 26 (for 15m in 6.5h day) -> 252 * 26 = 6552
        intervals_per_day = {
            '1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '1d': 1
        }
        steps = intervals_per_day.get(interval, 26)
        annual_factor = (252 * steps)**0.5
        current_vol = df['returns'].tail(20).std() * annual_factor
        if pd.isna(current_vol) or current_vol == 0:
            current_vol = 0.20 # Default

        df = features.compute_features(df)
        
        # 3. Predict on most recent candle
        if df.empty:
             return Response({'error': 'No data after features'}, status=500)
             
        current_data = df.iloc[[-1]].copy()
        
        # DEBUG: Print latest timestamp
        print(f"DEBUG: Prediction Time: {current_data.index[-1]}")
        print(f"DEBUG: Current Close: {current_data['Close'].iloc[0]}")

        # Get Probabilities
        probs = tm.predict_proba(current_data)[0] # [Bearish, Neutral, Bullish]
        
        # Determine Signal (Simple Argmax for now, or threshold)
        pred_idx = np.argmax(probs)
        signal_map = {0: 'PUT', 1: 'NEUTRAL', 2: 'CALL'}
        signal = signal_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        # 4. Calculate Levels
        entry_price = float(current_data['Close'].iloc[0])
        
        # Option Pricing Logic
        opt_engine = OptionsEngine()
        # Strike: Closest integer to price? Or just current price (ATM)
        strike = round(entry_price, 0)
        
        # DTE: Assume 0DTE logic (e.g. 0.5 day remaining) or use user input
        # Let's assume standard 0.4 day for "Next Trade"
        dte_days = 0.4 
        T = dte_days / 365.0
        r = 0.05
        
        opt_type = 'call' if signal == 'CALL' else 'put'
        
        if signal in ['CALL', 'PUT']:
            bs_res = opt_engine.black_scholes(S=entry_price, K=strike, T=T, r=r, sigma=current_vol, option_type=opt_type)
            option_entry = bs_res['price']
            
            # Target Options Prices
            # User Input: TP=500% (5.0), SL=10% (0.10)
            # These are usually on the OPTION Value
            option_tp = option_entry * (1 + take_profit_pct)
            option_sl = option_entry * (1 - stop_loss_pct)
            
            # For Visualization on Stock Chart:
            # We still want "approximate" stock levels where these would be hit.
            # Delta approximation: dOption = Delta * dStock
            # dStock = dOption / Delta
            delta = abs(bs_res['delta'])
            if delta < 0.01: delta = 0.01 # Prevent div by zero
            
            dOption_tp = option_tp - option_entry
            dStock_tp = dOption_tp / delta
            
            dOption_sl = option_entry - option_sl
            dStock_sl = dOption_sl / delta
            
            if signal == 'CALL':
                tp_price = entry_price + dStock_tp
                sl_price = entry_price - dStock_sl
            else: # PUT
                tp_price = entry_price - dStock_tp
                sl_price = entry_price + dStock_sl
        else:
            option_entry = 0.0
            option_tp = 0.0
            option_sl = 0.0
            tp_price = entry_price
            sl_price = entry_price
            strike = 0.0
            current_vol = 0.0
            
        # 5. Chart Data (Last 50 candles)
        recent_df = df.tail(50).copy()
        chart_data = recent_df[['Datetime', 'Close']].copy()
        chart_data['Datetime'] = chart_data['Datetime'].astype(str)
        
        return Response({
            'signal': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'option_data': {
                'entry': float(option_entry),
                'target': float(option_tp),
                'stop': float(option_sl),
                'strike': float(strike),
                'volatility': float(current_vol),
                'contract': f"{symbol} {signal} {strike} (Est.)"
            },
            'chart_data': chart_data.to_dict(orient='records'),
            'probs': {
                'bearish': float(probs[0]),
                'neutral': float(probs[1]),
                'bullish': float(probs[2])
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)
@api_view(['GET'])
def get_alpaca_account(request):
    try:
        from core.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        account = trader.get_account()
        if not account:
            return Response({'error': 'Could not fetch Alpaca account'}, status=500)
            
        return Response({
            'status': account.status,
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'currency': account.currency,
            'pattern_day_trader': account.pattern_day_trader,
            'daytrade_count': account.daytrade_count
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)
