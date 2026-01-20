from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Trade, Performance, ManualTrade
from .serializers import TradeSerializer, PerformanceSerializer, ManualTradeSerializer
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
    try:
        from core.alpaca_trader import AlpacaTrader
        from datetime import datetime
        
        trader = AlpacaTrader()
        
        # 1. Fetch Alpaca Data
        portfolio = trader.get_portfolio_history(period='1M', timeframe='1D')
        orders = trader.get_closed_orders(limit=50)
        
        equity_curve = []
        latest_trades = []
        stats = {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'current_equity': 1000.0
        }

        if portfolio:
            # Format Equity Curve
            # Portfolio object has lists: timestamp, equity
            timestamps = portfolio.timestamp
            equities = portfolio.equity
            
            for t, e in zip(timestamps, equities):
                # Convert timestamp (seconds) to datetime string
                dt = datetime.fromtimestamp(t)
                equity_curve.append({
                    'time': dt.strftime('%Y-%m-%d %H:%M'),
                    'equity': float(e) if e is not None else 0
                })
            
            if equity_curve:
                stats['current_equity'] = equity_curve[-1]['equity']
                
        else:
            # Fallback to DB logic if Alpaca fails
            pass 

        if orders is not None:
            # Format Orders
            wins = 0
            total_pnl = 0.0
            
            for o in orders:
                # Alpaca order object
                filled_at = o.filled_at
                if filled_at:
                    dt = filled_at.strftime('%Y-%m-%d %H:%M')
                else:
                    dt = o.created_at.strftime('%Y-%m-%d %H:%M')
                    
                # Calculate approximate PnL if possible, or just show price
                # Alpaca orders don't store PnL directly, we need meaningful display
                # For dashboard list:
                latest_trades.append({
                    'symbol': o.symbol,
                    'type': o.side.upper() if o.side else 'UNKNOWN',
                    'exit_time': dt,
                    'pnl_val': 0, # Alpaca doesn't give PnL per order easily without position tracking
                    'entry_price': float(o.filled_avg_price) if o.filled_avg_price else 0,
                    'qty': float(o.qty) if o.qty else 0
                })
            
            stats['total_trades'] = len(orders)
            # Cannot calculate win rate accurately from just orders list without knowing entry/exit pairs
            # But we can try to guess or leave it 0
            stats['win_rate'] = 0.0 
            
        else:
            # Fallback to DB logic if Alpaca fails or returns empty
             all_trades = Trade.objects.all().order_by('exit_time')
             total_trades = all_trades.count()
             if total_trades > 0:
                wins = all_trades.filter(pnl_val__gt=0).count()
                win_rate = (wins / total_trades) * 100.0
                total_pnl = sum(t.pnl_val for t in all_trades if t.pnl_val is not None)
                
                # Equity Curve Fallback
                if not equity_curve:
                    current_equity = 1000.0 
                    equity_curve.append({'time': 'Start', 'equity': current_equity})
                    for t in all_trades:
                        if t.exit_time and t.pnl_val is not None:
                            current_equity += t.pnl_val
                            equity_curve.append({
                                'time': t.exit_time.strftime('%Y-%m-%d %H:%M'),
                                'equity': current_equity
                            })
                            
                stats['total_trades'] = total_trades
                stats['win_rate'] = win_rate
                stats['total_pnl'] = total_pnl
                stats['current_equity'] = equity_curve[-1]['equity'] if equity_curve else 1000.0
             
             latest_trades = TradeSerializer(all_trades.order_by('-exit_time')[:50], many=True).data

        return Response({
            'latest_trades': latest_trades,
            'stats': stats,
            'equity_curve': equity_curve
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

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
                exit_reason=row.get('exit_reason', '')
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
        bullish_threshold = float(request.data.get('bullish_threshold', 70)) / 100.0
        bearish_threshold = float(request.data.get('bearish_threshold', 70)) / 100.0
        
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
        
        # Determine Signal based on thresholds
        # probs[0] = Bearish, probs[1] = Neutral, probs[2] = Bullish
        bearish_conf = float(probs[0])
        bullish_conf = float(probs[2])
        
        if bullish_conf >= bullish_threshold:
            signal = 'CALL'
            confidence = bullish_conf
        elif bearish_conf >= bearish_threshold:
            signal = 'PUT'
            confidence = bearish_conf
        else:
            signal = 'NEUTRAL'
            confidence = float(probs[1]) # Neutral confidence
        
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

@api_view(['POST'])
def run_manual_test(request):
    try:
        from core.manual_tester import ManualTester
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '15m')
        timestamp = request.data.get('timestamp') 
        
        if not timestamp:
            return Response({'error': 'Timestamp is required'}, status=400)
            
        tester = ManualTester(symbol, interval)
        pred = tester.run_prediction_at_time(timestamp)
        
        if not pred:
            return Response({'error': 'Could not run prediction at this time. Check data availability.'}, status=400)
            
        # Select OTM Option
        option_data = tester.select_otm_option(pred['entry_price'], pred['signal'], timestamp)
        
        return Response({
            'prediction': pred,
            'option': option_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def execute_auto_trade(request):
    try:
        from core.manual_tester import ManualTester
        from core.alpaca_trader import AlpacaTrader
        from datetime import datetime
        
        # 1. Parse Config
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '15m')
        risk_pct = float(request.data.get('risk_pct', 10)) / 100.0
        use_0dte = request.data.get('use0DTE', True)
        
        # 0. Check Market Status
        alpaca = AlpacaTrader()
        clock = alpaca.get_clock()
        if clock and not clock.is_open:
             # Override for manual testing if needed, but per user request, strictly "only when market open"
             return Response({'status': 'skipped', 'reason': 'Market is Closed'}, status=200)

        # 2. Run Prediction (Conceptually "Now")
        # Note: In a real live loop, we'd loop inside python. Here we rely on frontend poll.
        tester = ManualTester(symbol, interval)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # We need to ensure we have data. fetch_data defaults to end='now' approx.
        # run_prediction_at_time filters data <= timestamp.
        pred = tester.run_prediction_at_time(now_str)
        
        if not pred:
            return Response({'status': 'skipped', 'reason': 'No prediction data available/Market closed'}, status=200)
            
        signal = pred['signal']
        confidence = pred['confidence']
        
        # 3. Filter Signals
        if signal == 'NEUTRAL':
             return Response({'status': 'skipped', 'reason': 'Signal is NEUTRAL'}, status=200)
             
        # Add confidence threshold if needed (e.g., > 60%)
        if confidence < 0.60:
             return Response({'status': 'skipped', 'reason': f'Confidence too low ({confidence:.2f})'}, status=200)

        # 4. Select Option
        # We need the "Real" option price/contract for Alpaca
        # Since ManualTester simulates, we will try to construct a valid 0DTE symbol
        # and assume the "strike" selected is tradeable.
        
        timestamp_dt = pd.to_datetime(pred['timestamp'])
        option_data = tester.select_otm_option(pred['entry_price'], signal, timestamp_dt)
        
        if not option_data:
             return Response({'status': 'skipped', 'reason': 'Could not select valid option'}, status=200)
             
        strike = option_data['strike']
        
        alpaca = AlpacaTrader()
        
        # Construct Symbol
        # If 0DTE, expiration is Today
        today = datetime.now().date()
        # If market closed (e.g. now > 4pm ET), 0DTE exp is today (and invalid) or next day?
        # For simplicity, we assume "Next Trade" means Today if before close, or Next Day?
        # Let's assume testing during day -> Today.
        
        option_symbol = alpaca.get_option_contract(symbol, today, signal, strike)
        if not option_symbol:
             return Response({'error': 'Failed to construct option symbol'}, status=400)
             
        # 5. Calculate position size
        account = alpaca.get_account()
        if not account:
             return Response({'error': 'Failed to fetch Alpaca account'}, status=500)
             
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        
        # Allocated amount
        allocation = equity * risk_pct
        
        # Check option price (Use simulated premium as estimate or fetch quote?)
        # Fetching quote is safer but requires Data API.
        # We will use Market Order, so specific price not strictly needed for submission,
        # BUT needed for Quantity.
        # Estimate: option_data['premium'] from Black-Scholes (ManualTester)
        est_price = option_data['premium']
        
        if est_price <= 0:
             return Response({'status': 'skipped', 'reason': 'Estimated premium <= 0'}, status=400)
             
        # Qty = Allocation / (Price * 100)
        qty = int(allocation / (est_price * 100))
        
        if qty < 1:
             return Response({'status': 'skipped', 'reason': f'Insufficient capital for 1 contract (Alloc: ${allocation:.2f}, Cost: ${est_price*100:.2f})'}, status=200)
             
        # 6. Place Order
        # Check if already in position? (Simplification: No check, accumulated position)
        order = alpaca.place_order(
            symbol=option_symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        if order:
             return Response({
                 'status': 'executed',
                 'order_id': order.id,
                 'symbol': option_symbol,
                 'qty': qty,
                 'signal': signal,
                 'strike': strike,
                 'est_premium': est_price
             })
        else:
             return Response({'error': 'Alpaca order placement failed'}, status=500)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def save_manual_trade(request):
    try:
        serializer = ManualTradeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def get_manual_history(request):
    try:
        trades = ManualTrade.objects.all().order_by('-created_at')
        serializer = ManualTradeSerializer(trades, many=True)
        return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['DELETE'])
def clear_manual_history(request):
    try:
        ManualTrade.objects.all().delete()
        return Response({'status': 'success', 'message': 'Manual history cleared'})
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def run_auto_backtest(request):
    try:
        from core.manual_tester import ManualTester
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '15m')
        start_timestamp = request.data.get('timestamp')
        
        risk_pct = float(request.data.get('risk_pct', 10))
        stop_loss_pct = float(request.data.get('stop_loss_pct', 50))
        take_profit_pct = float(request.data.get('take_profit_pct', 200))
        position_size = float(request.data.get('position_size', 20))

        if not start_timestamp:
            return Response({'error': 'Start timestamp is required'}, status=400)

        # 1. Fetch data
        import data_loader
        df = data_loader.fetch_data(symbol, interval=interval, period='60d')
        if df.empty:
            return Response({'error': 'No data found'}, status=400)

        df['Datetime'] = pd.to_datetime(df['Datetime'])
        ts_start = pd.to_datetime(start_timestamp)
        
        # Sync timezone
        if df['Datetime'].dt.tz is not None and ts_start.tzinfo is None:
            ts_start = ts_start.tz_localize('UTC').tz_convert(df['Datetime'].dt.tz)
        elif df['Datetime'].dt.tz is None and ts_start.tzinfo is not None:
            ts_start = ts_start.replace(tzinfo=None)

        # Get data from start_timestamp onwards
        df_backtest = df[df['Datetime'] >= ts_start].copy()
        if len(df_backtest) < 2:
            return Response({'error': 'Not enough data for backtest from this point.'}, status=400)

        # 2. Generate signals for the entire period
        import features
        import model
        import signals
        
        tm = model.TradingModel(symbol=symbol, interval=interval)
        if not tm.load():
            # If no model, try training on full data (or just error out if we expect it to exist)
            df_full = features.compute_features(df)
            import patterns
            df_full = patterns.label_data(df_full)
            tm.train(df_full)
        
        df_all_features = features.compute_features(df)
        signals_df = signals.generate_signals(df_all_features, tm)
        
        # Filter signals to match our backtest period
        signals_df = signals_df[signals_df['Datetime'] >= ts_start].copy()

        # 3. Simplified Backtest Loop
        active_trade = None
        trades_saved = 0
        
        # Options Simulation Constants
        r = 0.05
        sigma = 0.20
        initial_hours_left = 12.0
        
        # Determine time sub per step
        intervals_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '1d': 1}
        steps_per_day = intervals_per_day.get(interval, 26)
        minutes_per_step = 390 / steps_per_day if interval != '1d' else 390
        
        from core.options_engine import OptionsEngine
        opt_engine = OptionsEngine()
        tester = ManualTester(symbol, interval)

        for i in range(len(signals_df)):
            row = signals_df.iloc[i]
            
            # Check for exits if trade is active
            if active_trade:
                steps_elapsed = i - active_trade['entry_index']
                current_hours_left = max(0, initial_hours_left - (steps_elapsed * minutes_per_step / 60))
                T = current_hours_left / (24 * 365)
                
                res = opt_engine.black_scholes(
                    S=row['Close'],
                    K=active_trade['option_strike'],
                    T=T,
                    r=r,
                    sigma=sigma,
                    option_type=active_trade['option_type'].lower()
                )
                
                curr_premium = res['price']
                
                result = None
                if curr_premium >= active_trade['tp_price']:
                    result = 'TP'
                elif curr_premium <= active_trade['sl_price']:
                    result = 'SL'
                elif i == len(signals_df) - 1:
                    result = 'Expired'
                
                if result:
                    # Finalize and Save
                    pnl_pct = (curr_premium - active_trade['option_premium']) / active_trade['option_premium']
                    # Mock balance for pnl_val (or just use 1000 as base)
                    pnl_val = (active_trade['position_size'] / 100.0 * 1000.0) * pnl_pct
                    
                    ManualTrade.objects.create(
                        symbol=symbol,
                        timestamp=active_trade['timestamp'],
                        prediction=active_trade['prediction'],
                        confidence=active_trade['confidence'],
                        option_strike=active_trade['option_strike'],
                        option_premium=active_trade['option_premium'],
                        option_type=active_trade['option_type'],
                        entry_price=active_trade['entry_price'],
                        take_profit=active_trade['tp_price'],
                        stop_loss=active_trade['sl_price'],
                        position_size=active_trade['position_size'],
                        result=result,
                        pnl_pct=pnl_pct * 100,
                        pnl_val=pnl_val
                    )
                    trades_saved += 1
                    active_trade = None

            # Check for entries if no active trade
            if not active_trade and row['signal'] in ['CALL', 'PUT']:
                # Select OTM option
                opt_data = tester.select_otm_option(row['Close'], row['signal'], row['Datetime'])
                if opt_data:
                    active_trade = {
                        'entry_index': i,
                        'timestamp': row['Datetime'],
                        'prediction': row['signal'],
                        'confidence': row['confidence'],
                        'option_strike': opt_data['strike'],
                        'option_premium': opt_data['premium'],
                        'option_type': row['signal'],
                        'entry_price': row['Close'],
                        'tp_price': opt_data['premium'] * (1 + take_profit_pct / 100),
                        'sl_price': opt_data['premium'] * (1 - stop_loss_pct / 100),
                        'position_size': position_size
                    }

        return Response({
            'status': 'success',
            'trades_processed': trades_saved,
            'message': f'Auto-backtest complete. Saved {trades_saved} trades to journal.'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def get_history_data(request):
    try:
        import data_loader
        symbol = request.GET.get('symbol', 'SPY')
        interval = request.GET.get('interval', '15m')
        period = request.GET.get('period', '60d')
        
        df = data_loader.fetch_data(symbol, interval=interval, period=period)
        if df.empty:
            return Response({'error': 'No data found'}, status=404)
            
        # Add index as timestamp string
        df_reset = df.reset_index()
        df_reset['Datetime'] = df_reset['Datetime'].astype(str)
        
        return Response(df_reset.to_dict(orient='records'))
    except Exception as e:
        return Response({'error': str(e)}, status=500)
@api_view(['GET'])
def health_check(request):
    status = {
        'backend': 'online',
        'market_status': 'unknown', 
        'next_open': None
    }
    try:
        from core.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        clock = trader.get_clock()
        if clock:
            status['market_status'] = 'open' if clock.is_open else 'closed'
            status['next_open'] = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        status['market_error'] = str(e)
        
    return Response(status)
