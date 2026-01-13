from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Trade, Performance
from .serializers import TradeSerializer, PerformanceSerializer
import sys
import os
import pandas as pd
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
            # Fetch and process data
            df = data_loader.fetch_data(symbol, interval=interval, period='60d')
            if df.empty:
                results[interval] = "No data"
                continue
                
            df = features.compute_features(df)
            df = patterns.label_data(df)
            
            # Train model
            tm = model.TradingModel()
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
    take_profit = float(request.data.get('take_profit', 500)) / 100.0
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
        
        tm = model.TradingModel()
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
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_journal.csv')
        if not os.path.exists(csv_path):
            return Response({'error': 'No trade details found (trade_journal.csv missing)'}, status=404)
            
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
@api_view(['POST'])
def get_prediction(request):
    try:
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '15m')
        stop_loss_pct = float(request.data.get('stop_loss', 10)) / 100.0
        take_profit_pct = float(request.data.get('take_profit', 500)) / 100.0
        
        # 1. Load Model
        tm = model.TradingModel()
        if not tm.load():
            return Response({'error': 'Model not found. Please train first.'}, status=400)
            
        # 2. Fetch Data (Need enough for features)
        df = data_loader.fetch_data(symbol, interval=interval, period='60d') # Need context for indicators
        if df.empty:
            return Response({'error': 'No data found'}, status=500)
            
        df = features.compute_features(df)
        
        # 3. Predict on most recent candle
        current_data = df.iloc[[-1]].copy()
        
        # Get Probabilities
        probs = tm.predict_proba(current_data)[0] # [Bearish, Neutral, Bullish]
        
        # Determine Signal (Simple Argmax for now, or threshold)
        pred_idx = np.argmax(probs)
        signal_map = {0: 'PUT', 1: 'NEUTRAL', 2: 'CALL'}
        signal = signal_map[pred_idx]
        confidence = float(probs[pred_idx])
        
        # 4. Calculate Levels
        entry_price = float(current_data['Close'].iloc[0])
        
        if signal == 'CALL':
            tp_price = entry_price * (1 + take_profit_pct * 0.05) # Scaling down T P/SL for visualization if needed, or use full? 
            # User wants visual chart. Usually TP is e.g. 1% move, so 500% TP meant options return, not underlying.
            # WAIT. The user form has TP% for OPTIONS (e.g. 500%).
            # But the chart shows UNDERLYING price.
            # We must convert Option TP% to Underlying Move %.
            # Approx: Delta=0.5 -> 1% stock move = 20% option move.
            # So 500% option move ~= 25% stock move. That's huge for 15m chart.
            # Let's use a "Visual" TP/SL based on recent volatility (ATR) or fixed small % for the predictor chart demo, 
            # OR ask the user for "Target Stock Price". 
            # Let's infer a "Projected Move" based on ATR or simply a fixed ratio for visualization.
            # Actually, standard stop loss is often 10% on option -> ~0.5% on stock.
            # Let's estimate: Option leverage ~20-50x. 
            # Let's start with a synthetic underlying moves for visualization: SL=-0.5%, TP=1.5%?
            # Or use the user's "Stop Loss" input but treat it as Option SL.
            
            # Let's derive nice levels for the chart based on current price volatility
            # A good visual target: +2*ATR for TP, -1*ATR for SL.
            # But I should try to honor the user's risk ratio.
            rr_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 2
            
            # Visual SL distance: let's say 0.5% of price
            sl_move = 0.005 
            tp_move = sl_move * (rr_ratio / 5) # Scale it a bit so it fits chart
            
            tp_price = entry_price * (1 + tp_move)
            sl_price = entry_price * (1 - sl_move)
            
        elif signal == 'PUT':
            rr_ratio = take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 2
            sl_move = 0.005 
            tp_move = sl_move * (rr_ratio / 5)
            
            tp_price = entry_price * (1 - tp_move)
            sl_price = entry_price * (1 + sl_move)
            
        else:
            tp_price = entry_price
            sl_price = entry_price
            
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
