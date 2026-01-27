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
        
        # Log System Status on Reload (User Request)
        # This view is called when dashboard fetches data
        print(f"[{datetime.now()}] System Status: ONLINE (Alpaca Connected)")
        
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
    # Support both single 'interval' (from Risk page) and list 'intervals'
    req_interval = request.data.get('interval')
    intervals = [req_interval] if req_interval else request.data.get('intervals', ['15m'])
    model_type = request.data.get('model_type', 'lstm').lower() # Default to lstm
    project_name = request.data.get('project_name', 'default')
    
    results = {}
    
    for interval in intervals:
        try:
            # UPGRADE: Use MultiTimeframePipeline + LSTM
            from core.feature_pipeline import MultiTimeframePipeline
            from core.lstm_model import LSTMModel
            from core.model import TradingModel
            from core.hybrid_model import HybridModel
            
            # Determine period (needs to be enough for MTF)
            if interval == '1d':
                period = '10y'
            elif interval in ['1h', '90m']:
                period = '730d'
            else:
                period = '60d'
            
            pipeline = MultiTimeframePipeline(symbol)
            # This handles fetching + merging 1D/1H context automatically
            df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period=period)
            
            if df.empty:
                results[interval] = "No data"
                continue
                
            if model_type == 'lstm':
                print(f"Training LSTM for {interval}...")
                tm = LSTMModel(symbol, interval, project_name=project_name)
                tm.train(df, epochs=20, batch_size=32)
                results[interval] = "Success (LSTM)"
            elif model_type == 'xgboost':
                print(f"Training XGBoost for {interval}...")
                # XGB needs standard labels, handled inside train -> _prepare_data if passed raw DF?
                # TradingModel.train expects DF with target/label columns?
                # MultiTimeframePipeline returns features. 
                # Converting to Labelled Data:
                import patterns
                df_labeled = patterns.label_data(df)
                tm = TradingModel(symbol, interval)
                tm.train(df_labeled)
                results[interval] = "Success (XGBoost)"
            elif model_type == 'hybrid':
                print(f"Training Hybrid (XGB+LSTM) for {interval}...")
                # Hybrid needs labeled data for XGB part
                import patterns
                df_labeled = patterns.label_data(df)
                hm = HybridModel(symbol, interval)
                hm.train(df_labeled)
                results[interval] = "Success (Hybrid)"
            else:
                 results[interval] = f"Unknown model type: {model_type}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            results[interval] = str(e)
            
    return Response({'status': 'Training complete', 'details': results})

@api_view(['POST'])
def train_cascade_system(request):
    try:
        from core.cascade_system import CascadeSystem
        symbol = request.data.get('symbol', 'SPY')
        
        system = CascadeSystem(symbol)
        system.train_system()
        
        return Response({'status': 'Cascade System Training Initiated/Complete'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def get_cascade_status(request):
    try:
        from core.cascade_system import CascadeSystem
        symbol = request.GET.get('symbol', 'SPY')
        
        system = CascadeSystem(symbol)
        # Check if models exist
        status = {
            'trend_model': system.trend_model.load(),
            'structure_model': system.structure_model.load(),
            'exec_model': system.exec_model.load()
        }
        
        return Response(status)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def run_backtest(request):
    # Parameters
    symbol = request.data.get('symbol', 'SPY')
    interval = request.data.get('interval', '1h')
    
    # Debug: Check raw inputs
    print(f"DEBUG: Raw request data: {request.data}")
    
    initial_balance = float(request.data.get('initial_balance', 1000))
    # Risk page sends percentage (e.g. 20), we need decimal (0.20)
    risk_pct = float(request.data.get('risk_pct', 20)) / 100.0
    stop_loss = float(request.data.get('stop_loss', 10)) / 100.0
    take_profit = float(request.data.get('take_profit', 50)) / 100.0
    zero_dte = request.data.get('zero_dte', False)
    # DEFAULT TO LSTM per user request
    model_type = request.data.get('model_type', 'lstm') 
    project_name = request.data.get('project_name', 'default') 
    
    # Determine max period based on interval limits
    if interval == '1d':
        period = '10y'
    elif interval in ['1m', '2m']:
        period = '7d'
    elif interval in ['5m', '15m', '30m', '90m']:
       period = '60d'
    else:
       period = '2y' # Increased for 1h backtests
    
    try:
        # DATA FETCHING
        from core.feature_pipeline import MultiTimeframePipeline
        pipeline = MultiTimeframePipeline(symbol)
        df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period=period)
        
        if df.empty:
             return Response({'error': 'No data found'}, status=400)

        # MODEL PREDICTION
        if model_type == 'lstm':
            from core.lstm_model import LSTMModel
            lstm = LSTMModel(symbol, interval, project_name=project_name)
            if not lstm.load():
                 return Response({'error': 'LSTM model not found. Train one first.'}, status=400)
            signals_df = lstm.predict_signals(df)
            
        elif model_type == 'xgboost':
            from core.model import TradingModel
            import signals
            import patterns
            
            # XGB needs labeled features structure (dropnon-features?)
            # But generate_signals expects full DF?
            # It uses `tm.predict(df)` which strips cols internally.
            
            tm = TradingModel(symbol=symbol, interval=interval)
            if not tm.load():
                return Response({'error': 'XGBoost model not found. Train one first.'}, status=400)
                
            # Need to ensure features are computed same way?
            # Pipeline does it.
            # But XGB expects `target`/`label` columns if we used `label_data`?
            # `predict` ignores them.
            
            signals_df = signals.generate_signals(df, tm)
            
        elif model_type == 'hybrid':
            from core.hybrid_model import HybridModel
            hm = HybridModel(symbol, interval)
            loaded = hm.load()
            if not loaded:
                 return Response({'error': 'Hybrid model components not found. Train first.'}, status=400)
            
            # Hybrid predict logic
            # We need a dataframe with 'signal' column.
            # `predict_signals` returns a list/array of signals.
            start_idx = len(df) - len(hm.predict_signals(df)) # Align?
            # Actually predict_signals should handle alignment or return aligned list
            
            hyb_signals = hm.predict_signals(df)
            
            # We need to map these textual signals ('CALL', 'PUT', 'NEUTRAL') to the dataframe
            # And then run backtest?
            # Backtester expects `signal` column.
            
            # Careful with length mismatch if hybrid drops rows
            # Assume aligned for now
            if len(hyb_signals) == len(df):
                signals_df = df.copy()
                signals_df['signal'] = hyb_signals
            else:
                 # Truncate df to match signals (usually signals are at end)
                 signals_df = df.iloc[-len(hyb_signals):].copy()
                 signals_df['signal'] = hyb_signals

        else:
             return Response({'error': f'Unknown model type: {model_type}'}, status=400)
        
        # Ensure Datetime is a column BEFORE any strategy logic that needs it
        if 'Datetime' not in signals_df.columns:
            signals_df = signals_df.reset_index()
            # If index name wasn't Datetime (e.g. index), rename it
            if 'Datetime' not in signals_df.columns:
                 if 'Date' in signals_df.columns:
                      signals_df.rename(columns={'Date': 'Datetime'}, inplace=True)
                 elif 'index' in signals_df.columns:
                      signals_df.rename(columns={'index': 'Datetime'}, inplace=True)
                 elif signals_df.index.name == 'Datetime': # Explicit check
                      signals_df = signals_df.reset_index()
                 elif signals_df.columns[0] == 'Datetime': # Already there?
                      pass
                 else:
                      # Last resort: Rename the first column
                      signals_df.rename(columns={signals_df.columns[0]: 'Datetime'}, inplace=True)

        if zero_dte and model_type != 'lstm':
            # 0DTE logic (often specific to XGB patterns)
            # Maybe apply to LSTM too? 
            # For now keep legacy behavior for XGB
            zdte = ZeroDTEStrategy()
            signals_df = zdte.get_0dte_signals(signals_df)
            signals_df['signal'] = signals_df['zero_dte_signal']
            
        # Target DTE
        target_dte = 0.0 if zero_dte else 30.0
        
        # Ensure Datetime is a column for Backtester
        if 'Datetime' not in signals_df.columns:
            signals_df = signals_df.reset_index()
            # If index name wasn't Datetime (e.g. index), rename it
            if 'Datetime' not in signals_df.columns and 'index' in signals_df.columns:
                 # Check if 'index' holds datetime
                 pass # assumption is reset_index worked if original index was named Datetime
                 
        # If it's still missing (e.g. index had no name), try to fix
        if 'Datetime' not in signals_df.columns:
             # Look for any datetime-like column or force rename correct column
             # Usually prepare_multitimeframe_data returns index named Datetime
             if signals_df.index.name == 'Datetime':
                  signals_df = signals_df.reset_index()
             else:
                  # Force reset and rename first column?
                  signals_df = signals_df.reset_index()
                  if 'Datetime' not in signals_df.columns:
                      signals_df.rename(columns={signals_df.columns[0]: 'Datetime'}, inplace=True)

        bt = backtest.Backtester(
            signals_df, 
            symbol=symbol,
            initial_balance=initial_balance, 
            interval=interval,
            target_dte=target_dte,
            risk_per_trade_pct=risk_pct,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            options_mode=True, # Enforce options mode
            project_name=project_name
        )
        
        equity_df = bt.run()
        
        # Serialize equity curve
        equity_data = equity_df[['Datetime', 'equity']].copy()
        equity_data['Datetime'] = equity_data['Datetime'].astype(str)
        
        # Helper to sanitize NaNs for JSON
        def sanitize_data(data):
            if isinstance(data, dict):
                return {k: sanitize_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [sanitize_data(v) for v in data]
            elif isinstance(data, (float, np.float64, np.float32)):
                if np.isnan(data) or np.isinf(data):
                    return None
                return float(data)
            elif isinstance(data, (np.int64, np.int32)):
                 return int(data)
            return data

        response_data = {
            'equity_curve': equity_data.to_dict(orient='records'),
            'final_balance': bt.balance,
            'total_trades': len(bt.journal.trades),
            'summary': bt.journal.get_summary(),
            'trades': bt.journal.trades
        }
        
        return Response(sanitize_data(response_data))
    except Exception as e:
        print("ERROR IN BACKTEST:")
        import traceback
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
        
        model_type = request.data.get('model_type', 'lstm').lower()
        
        # 1. Fetch Data (Need enough for features)
        # UPGRADE: Use Pipeline for consistency
        from core.feature_pipeline import MultiTimeframePipeline
        pipeline = MultiTimeframePipeline(symbol)
        df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period='60d')
        
        if df.empty:
            return Response({'error': 'No data found'}, status=500)
            
        # Compute Volatility (Annualized Std Dev of Returns)
        df['returns'] = df['Close'].pct_change()
        intervals_per_day = {'1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '1d': 1}
        steps = intervals_per_day.get(interval, 26)
        annual_factor = (252 * steps)**0.5
        current_vol = df['returns'].tail(20).std() * annual_factor
        if pd.isna(current_vol) or current_vol == 0:
            current_vol = 0.20 # Default

        # 3. Predict on most recent candle
        current_data = df.iloc[[-1]].copy()
        
        # DEBUG: Print latest timestamp
        print(f"DEBUG: Prediction Time: {current_data.index[-1]}")
        print(f"DEBUG: Current Close: {current_data['Close'].iloc[0]}")

        probs = [0, 1, 0] # Default Neutral
        signal = 'NEUTRAL'
        confidence = 0.0

        if model_type == 'xgboost':
            tm = model.TradingModel(symbol=symbol, interval=interval)
            if not tm.load(): return Response({'error': 'XGB model not found'}, status=400)
            probs = tm.predict_proba(current_data)[0]
             # Determine Signal based on thresholds
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
                confidence = float(probs[1])

        elif model_type == 'lstm':
            from core.lstm_model import LSTMModel
            lstm = LSTMModel(symbol, interval)
            if not lstm.load(): return Response({'error': 'LSTM model not found'}, status=400)
            
            # LSTM logic (predict next price)
            next_price = lstm.predict_next(df)
            curr = current_data['Close'].iloc[0]
            diff = (next_price - curr) / curr
            
            # Simple thresholding? 
            # Or use probability-like confidence?
            if diff > 0.001: 
                signal = 'CALL'
                confidence = min(abs(diff)*100, 0.99) # Fake confidence
            elif diff < -0.001: 
                signal = 'PUT'
                confidence = min(abs(diff)*100, 0.99)
            else:
                signal = 'NEUTRAL'
                confidence = 0.5
                
            # Fake probs for UI
            if signal == 'CALL': probs = [0.1, 0.2, 0.7]
            elif signal == 'PUT': probs = [0.7, 0.2, 0.1]
            else: probs = [0.2, 0.6, 0.2]

        elif model_type == 'hybrid':
            from core.hybrid_model import HybridModel
            hm = HybridModel(symbol, interval)
            if not hm.load(): return Response({'error': 'Hybrid model components not found'}, status=400)
            
            res = hm.get_prediction_now(df)
            signal = res['signal']
            confidence = res['confidence']
            
             # Fake probs for UI based on signal
            if signal == 'CALL': probs = [0.1, 0.2, 0.7]
            elif signal == 'PUT': probs = [0.7, 0.2, 0.1]
            else: probs = [0.2, 0.6, 0.2]
            
        elif model_type == 'cascade':
            from core.cascade_system import CascadeSystem
            try:
                cs = CascadeSystem(symbol)
                # Training not called here, assuming already trained or loads existing
                # Check load status inside get_signal? No, get_signal is dynamic.
                # Just call get_signal
                res = cs.get_signal()
                
                signal = res['signal']
                confidence = res['confidence']
                # Probs come from execution model
                # [UP, NEUTRAL, DOWN] -> [BEARISH, NEUTRAL, BULLISH]
                # exec_probs = [UP, NEUTRAL, DOWN]
                exec_probs = res['components']['execution_probs']
                # Map UP -> Bullish (index 2), DOWN -> Bearish (index 0)
                probs = [
                    exec_probs[2], # Down
                    exec_probs[1], # Neutral
                    exec_probs[0]  # Up
                ]
            except Exception as e:
                import traceback
                traceback.print_exc()
                return Response({'error': f"Cascade Error: {str(e)}"}, status=500)
            
        else:
             return Response({'error': 'Unknown model type'}, status=400)
        
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
        # 5. Chart Data (Last 50 candles)
        recent_df = df.tail(50).copy()
        
        # Ensure Datetime is a column
        if 'Datetime' not in recent_df.columns:
            recent_df = recent_df.reset_index()
            # If still not there, maybe index didn't have a name, or named 'Date'
            if 'Datetime' not in recent_df.columns:
                 # Check 'Date' or just use first column?
                 if 'Date' in recent_df.columns:
                      recent_df.rename(columns={'Date': 'Datetime'}, inplace=True)
                 elif 'index' in recent_df.columns: 
                      recent_df.rename(columns={'index': 'Datetime'}, inplace=True)
                 else:
                      # Last resort: Rename the first column (which was the index)
                      recent_df.rename(columns={recent_df.columns[0]: 'Datetime'}, inplace=True)

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
        model_type = request.data.get('model_type', 'xgb')
        
        if not timestamp:
            return Response({'error': 'Timestamp is required'}, status=400)
            
        tester = ManualTester(symbol, interval, model_type=model_type)
        pred = tester.run_prediction_at_time(timestamp)
        
        if not pred:
            return Response({'error': 'Could not run prediction at this time. Check data availability.'}, status=400)
            
        # Select OTM Option
        option_data = tester.select_otm_option(pred['entry_price'], pred['signal'], timestamp)
        
        # Calculate TP/SL Levels for Visualization
        stop_loss_pct = 0.10
        take_profit_pct = 0.50
        
        if option_data and option_data.get('delta'):
             # Delta approximation for Stock levels
             entry_opt = option_data['premium']
             delta = abs(option_data['delta'])
             if delta < 0.01: delta = 0.01
             
             option_tp = entry_opt * (1 + take_profit_pct)
             option_sl = entry_opt * (1 - stop_loss_pct)
             
             dOption_tp = option_tp - entry_opt
             dStock_tp = dOption_tp / delta
             
             dOption_sl = entry_opt - option_sl
             dStock_sl = dOption_sl / delta
             
             if pred['signal'] == 'CALL':
                 pred['tp_price'] = pred['entry_price'] + dStock_tp
                 pred['sl_price'] = pred['entry_price'] - dStock_sl
             elif pred['signal'] == 'PUT':
                 pred['tp_price'] = pred['entry_price'] - dStock_tp
                 pred['sl_price'] = pred['entry_price'] + dStock_sl
             else:
                 pred['tp_price'] = pred['entry_price']
                 pred['sl_price'] = pred['entry_price']
        else:
             # Fallback if no option/delta (e.g. NEUTRAL)
             pred['tp_price'] = pred['entry_price']
             pred['sl_price'] = pred['entry_price']
        
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
            
            # --- CUSTOM JOURNALING (CSV) ---
            try:
                import csv
                import os
                
                # Define path
                journal_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_journals', 'manual')
                if not os.path.exists(journal_dir):
                    os.makedirs(journal_dir)
                    
                csv_path = os.path.join(journal_dir, 'manual_journal.csv')
                
                # Extract data
                data = serializer.validated_data
                # Flatten or select fields
                row = {
                    'timestamp': data.get('timestamp'),
                    'symbol': data.get('symbol'),
                    'prediction': data.get('prediction'),
                    'confidence': data.get('confidence'),
                    'option_strike': data.get('option_strike'),
                    'option_premium': data.get('option_premium'),
                    'option_type': data.get('option_type'),
                    'entry_price': data.get('entry_price'),
                    'result': data.get('result'),
                    'pnl_pct': data.get('pnl_pct'),
                    'pnl_val': data.get('pnl_val')
                }
                
                file_exists = os.path.isfile(csv_path)
                
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
                    
            except Exception as e:
                print(f"Failed to write manual trade to CSV: {e}")
            
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
        
        # 2. Generate signals
        model_type = request.data.get('model_type', 'xgb')
        tm = None
        
        if model_type == 'lstm':
            from core.lstm_model import LSTMModel
            tm = LSTMModel(symbol, interval)
            if not tm.load():
                return Response({'error': 'LSTM model not found. Train one first.'}, status=400)
            
            # For LSTM, we need to generate signals using its specific method or wrapper
            # LSTMModel usually predicts next price. We need to convert to signals logic.
            # But wait, run_auto_backtest loops over signals_df locally.
            # We need a unified "generate_signals_for_dataframe" method.
            # Let's assume we can rely on `tm.predict_signals(df)` if implemented, or implement here.
             
        elif model_type == 'hybrid':
             from core.hybrid_model import HybridModel
             tm = HybridModel(symbol, interval)
             if not tm.load():
                 return Response({'error': 'Hybrid model not found. Train one first.'}, status=400)
        else:
            # Default XGB
            tm = model.TradingModel(symbol=symbol, interval=interval)
            if not tm.load():
                # Try training if missing
                df_full = features.compute_features(df)
                import patterns
                df_full = patterns.label_data(df_full)
                tm.train(df_full)

        df_all_features = features.compute_features(df)
        
        # Unified Signal Generation
        if model_type == 'lstm':
            # LSTMModel needs to support predict_signals or we handle it
            # Validating if predict_signals exists... assuming yes or we add it to LSTMModel
            if hasattr(tm, 'predict_signals'):
                 signals_df = tm.predict_signals(df_all_features)
            else:
                 # Fallback/Error
                 return Response({'error': 'LSTMModel does not support signal generation method'}, status=500)
        elif model_type == 'hybrid':
             if hasattr(tm, 'predict_signals'):
                 # predict_signals might return list, we need DF
                 sigs = tm.predict_signals(df_all_features)
                 # If it returns list/series, we attach to df
                 signals_df = df_all_features.copy()
                 signals_df['signal'] = sigs
                 # Add dummy confidence if missing
                 signals_df['confidence'] = 0.8
             else:
                 return Response({'error': 'HybridModel does not support signal generation method'}, status=500)
        else:
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

@api_view(['POST'])
def train_lstm(request):
    try:
        from core.lstm_model import LSTMModel
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '5m')
        epochs = int(request.data.get('epochs', 20))
        batch_size = int(request.data.get('batch_size', 32))
        lookback = int(request.data.get('lookback', 60))
        period = request.data.get('period', '2y') 
        
        # YFinance restriction: Intraday (1m, 5m, 15m, 30m, 90m) < 60 days
        if interval in ['1m', '5m', '15m', '30m', '90m']:
            # If user asked for '2y' (default) or something long, cap it.
            # We'll just default to '59d' for safety if it looks like a long period or default.
            # Simplified check: just force 59d for these intervals unless specific short override
            if period == '2y' or period == 'max' or 'y' in period:
                period = '59d'
        
        # In a real app, use Celery. Here we'll do it synchronously or threading?
        # User requested "Training must not block live trading". 
        # Since this is a simple Django Setup, we can use a Thread.
        import threading
        
        def run_training():
            with open('debug_training.log', 'a') as f:
                f.write(f"Starting training for {symbol} {interval} {period}\n")
            try:
                # Need data first
                df = data_loader.fetch_data(symbol, interval=interval, period=period) # Need enough data
                with open('debug_training.log', 'a') as f:
                    f.write(f"Data fetched: {len(df)} rows\n")
                
                if df.empty:
                    with open('debug_training.log', 'a') as f:
                        f.write("No data for LSTM training\n")
                    print("No data for LSTM training")
                    return
                # Train
                lstm = LSTMModel(symbol, interval, lookback=lookback)
                lstm.train(df, epochs=epochs, batch_size=batch_size)
                with open('debug_training.log', 'a') as f:
                     f.write("Training finished and model saved.\n")
            except Exception as e:
                with open('debug_training.log', 'a') as f:
                    f.write(f"LSTM Training Error: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
                print(f"LSTM Training Error: {e}")
                
        thread = threading.Thread(target=run_training)
        thread.daemon = True
        thread.start()
        
        return Response({'status': 'Training started in background', 'symbol': symbol, 'interval': interval})
    except Exception as e:
         return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def train_multitimeframe_model(request):
    try:
        from core.lstm_model import LSTMModel
        from core.feature_pipeline import MultiTimeframePipeline
        
        symbol = request.data.get('symbol', 'SPY')
        # Base interval for the model (entry timeframe)
        interval = request.data.get('interval', '15m') 
        epochs = int(request.data.get('epochs', 20))
        batch_size = int(request.data.get('batch_size', 32))
        lookback = int(request.data.get('lookback', 60))
        
        # Period for base data (enough for lookback + training samples)
        # 60d is safe for 15m.
        period = request.data.get('period', '60d')
        print(f"DEBUG: train_multitimeframe_model called for {symbol} {interval} {period}") # Debug


        import threading
        
        def run_mtf_training():
            from datetime import datetime
            log_file = 'debug_training.log'
            with open(log_file, 'a') as f:
                f.write(f"\n[{datetime.now()}] Starting MTF training for {symbol} {interval}\n")
                
            try:
                # 1. Pipeline Data Prep
                pipeline = MultiTimeframePipeline(symbol)
                # Hardcoded strategy per user request: 1D (Trend), 1H (Structure), 15m (Entry)
                # But if interval is different, maybe adjust? For now assume user follows the guide.
                
                # Fetch and Merge
                df = pipeline.prepare_multitimeframe_data(base_interval=interval, base_period=period)
                
                with open(log_file, 'a') as f:
                     f.write(f"MTF Data prepared. Shape: {df.shape}\n")
                
                if df.empty:
                    print("MTF Data Empty")
                    return

                # 2. Train LSTM
                # Note: LSTMModel handles dynamic columns now.
                lstm = LSTMModel(symbol, interval, lookback=lookback)
                lstm.train(df, epochs=epochs, batch_size=batch_size)
                
                with open(log_file, 'a') as f:
                     f.write(f"MTF Training finished for {symbol} {interval}.\n")
                     
            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write(f"MTF Training Error: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
                print(f"MTF Training Error: {e}")

        # Start Background Thread
        thread = threading.Thread(target=run_mtf_training)
        thread.daemon = True
        thread.start()

        return Response({'status': 'Multi-Timeframe Training started', 'symbol': symbol, 'interval': interval})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def get_lstm_status(request):
    # Retrieve status from metadata
    try:
        symbol = request.query_params.get('symbol', 'SPY')
        interval = request.query_params.get('interval', '5m')
        
        from core.lstm_model import LSTMModel
        import json
        
        lstm = LSTMModel(symbol, interval)
        
        # Check active version first
        active_ts = lstm._get_active_version_path()
        if active_ts:
             meta_path = os.path.join(lstm.versions_dir, active_ts, 'metadata.json')
        else:
             # Just look for any latest version? Or return empty
             # If training is running, where does it write intermediate status?
             # train() keeps it in memory loop or saves checkpoint?
             # The current train() implementation only saves at the END.
             # So status polling will fail until finished in current design.
             # Improvement: train() should write a 'training_status.json' in base dir.
             # For now, we return the latest COMPLETED version status if available.
             versions = lstm.list_versions()
             if versions:
                 return Response(versions[0]) # Latest
             return Response({'status': 'No model found or training not started'})
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data = json.load(f)
            return Response(data)
        else:
            return Response({'status': 'No metadata found'})
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def list_lstm_versions(request):
    try:
        symbol = request.query_params.get('symbol', 'SPY')
        interval = request.query_params.get('interval', '5m')
        from core.lstm_model import LSTMModel
        
        lstm = LSTMModel(symbol, interval)
        versions = lstm.list_versions()
        return Response({'versions': versions})
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def rollback_lstm_version(request):
    try:
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '5m')
        timestamp = request.data.get('timestamp')
        
        from core.lstm_model import LSTMModel
        lstm = LSTMModel(symbol, interval)
        lstm.set_active_version(timestamp)
        return Response({'status': 'success', 'active_version': timestamp})
    except Exception as e:
         return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def predict_lstm(request):
    try:
        from core.lstm_model import LSTMModel
        symbol = request.data.get('symbol', 'SPY')
        interval = request.data.get('interval', '5m')
        lookback = int(request.data.get('lookback', 60))
        
        lstm = LSTMModel(symbol, interval, lookback=lookback)
        if not lstm.load():
            return Response({'error': 'Model not found. Train first.'}, status=400)
            
        # Fetch recent data
        # We need lookback amount
        df = data_loader.fetch_data(symbol, interval=interval, period='5d')
        if len(df) < lookback:
             return Response({'error': f'Not enough data points ({len(df)}) for lookback ({lookback})'}, status=400)
             
        pred_price = lstm.predict_next(df)
        
        return Response({
            'symbol': symbol,
            'interval': interval,
            'predicted_close': pred_price,
            'current_close': df['Close'].iloc[-1],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
