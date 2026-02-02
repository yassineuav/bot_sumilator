import pandas as pd
import numpy as np

try:
    from .data_loader import fetch_data
    from .features import compute_features
except ImportError:
    # Fallback for when core is in sys.path but module is top-level
    try:
        from data_loader import fetch_data
        from features import compute_features
    except ImportError:
        # Fallback for when backend is in sys.path
        from core.data_loader import fetch_data
        from core.features import compute_features

class MultiTimeframePipeline:
    def __init__(self, symbol):
        self.symbol = symbol
        
    def fetch_and_process_higher_tf(self, interval, period, shift_lag=1):
        """
        Fetches data for a higher timeframe, computes features, and prepares it for merging.
        shift_lag: Number of intervals to shift to avoid lookahead bias. 
                   For 1D features used in intraday, we want yesterday's data known at today's open.
                   So shift=1 means row T's features are from T-1.
        """
        df = fetch_data(self.symbol, interval=interval, period=period)
        if df.empty:
            print(f"Warning: No data for {self.symbol} {interval}")
            return pd.DataFrame()

        # Ensure Datetime Index FIRST
        if 'Datetime' in df.columns:
            df = df.set_index('Datetime')
            
        # Compute generic features (RSI, MACD, Moving Averages)
        # We rely on core.features.compute_features but might need custom ones for "Regime"
        df = compute_features(df)
        
        # Add Timeframe Prefix to COLUMNS only
        df.columns = [f"{interval}_{c}" for c in df.columns]
        
        # Shift to avoid lookahead bias
        # If we are at 10:00, we know the 9:00 close (if 1h bars).
        # But yfinance timestamps are usually start of bar. 
        # 10:00 bar closes at 11:00.
        # If we test at 10:15, we technically don't know the 10:00 bar close yet?
        # Wait, if 1H bar is 09:30-10:30. At 10:15 we only know 09:30 bar (which ended 10:30?? No).
        # Standard Exchange Hours: 9:30, 10:30, 11:30...
        # Canvas:
        # 1D: Today is T. We know T-1.
        # 1H: Current time T_now. We know the fully completed hourly candle before T_now.
        
        # Simple approach: Shift by 1 row.
        # When we resample/reindex to finer timeframe, we rely on 'ffill'.
        # If we shift by 1 here, the row at index T contains data from T-1.
        # Then ffill will propagate T-1 data forward until T updates.
        
        df_shifted = df.shift(shift_lag)
        
        # We need to keep the index (Datetime) unshifted so we can merge on time?
        # NO. If we shift the data but keep index, then at time T we see data from T-1. Correct.
        return df_shifted

    def prepare_multitimeframe_data(self, base_interval='15m', base_period='60d'):
        """
        Main orchestration function.
        1. Fetch 1D (Trend) - 2y (sufficient for 100-200 MA)
        2. Fetch 1H (Structure) - 1y
        3. Fetch 15m (Entry) - 60d
        4. Merge
        """

        print(f"Preparing Multi-Timeframe Data for {self.symbol}...", flush=True)
        
        # 1. Base Data (Target)
        base_df = fetch_data(self.symbol, interval=base_interval, period=base_period)
        if base_df.empty:
            raise ValueError(f"No base data for {self.symbol} {base_interval}")
            
        base_df = compute_features(base_df)
        # Keep base columns as is, or maybe prefix? Let's keep distinct.
        
        # Ensure Datetime index 
        if 'Datetime' in base_df.columns:
            base_df = base_df.set_index('Datetime')
        # If index is already Datetime, good.
        # Ensure localized awareness if needed, usually yfinance is tz-aware.
        
        # 2. 1D Data (Trend)
        # Need enough history for 200 SMA. 10y or max is preferred per user request.
        df_1d = self.fetch_and_process_higher_tf('1d', '10y')
        if not df_1d.empty:
            # df_1d already has Datetime Index from fetch_and_process_higher_tf
            
            # Merge 1D
            # Option 1: Join via Date.
            # Option 2: Reindex with ffill (simpler for 1D to Intraday expansion)
            # Since df_1d is daily (00:00 usually), and we want it to apply to all intraday bars of that day.
            # We already shifted it by 1.
            # So Row(Today) has Yesterday's Data.
            # If we forward fill, all Intraday(Today) will inherit Row(Today) -> Yesterday's Data. Correct.
            
            # Reindex 1D to match Base Index
            # First, we need to ensure the index covers the range.
            # merge_asof is safer than reindex sometimes, but reindex(method='ffill') is classic for timeframe expansion.
            
            base_df = base_df.sort_index()
            df_1d = df_1d.sort_index()
            
            # Ensure index types match (avoid resolution mismatch e.g. s vs ns)
            if base_df.index.dtype != df_1d.index.dtype:
                df_1d.index = df_1d.index.astype(base_df.index.dtype)
            
            # Use merge_asof backwards.
            # For 1D data at T=00:00 (Today).
            # Intraday T=09:30 (Today). 
            # backward search finds T=00:00.
            # So it picks up Today's row.
            # Today's row was shifted to contain Yesterday's data.
            # So 09:30 picks up Yesterday's data. Correct.
            
            merged = pd.merge_asof(
                base_df,
                df_1d,
                left_index=True,
                right_index=True,
                direction='backward', # matches < 09:30 which is 00:00
                suffixes=('', '_1d')
            )
            base_df = merged
            
        # 3. 1H Data (Momentum/Structure)
        df_1h = self.fetch_and_process_higher_tf('1h', '730d') # 2y max for hourly
        if not df_1h.empty:
             if 'Datetime' in df_1h.columns: df_1h = df_1h.set_index('Datetime')
             
             # Merge 1H
             # 1H timestamps: 9:30, 10:30...
             # Base 15m: 9:30, 9:45, 10:00, 10:15...
             # We shifted 1H by 1.
             # So 1H(10:30) row has data from 9:30 bar.
             
             # Resample/Merge
             # We can use merge_asof if sorted.
             base_df = base_df.sort_index()
             df_1h = df_1h.sort_index()
             
             # Ensure index types match
             if base_df.index.dtype != df_1h.index.dtype:
                 df_1h.index = df_1h.index.astype(base_df.index.dtype)
             
             # merge_asof matches nearest backward.
             # If base is 10:15, and we have 1H at 9:30 and 10:30.
             # We want the data available at 10:15.
             # The most recent CLOSED hourly bar is 9:30 (closes 10:30? No).
             # Yfinance 1h starts at 9:30. Ends 10:30.
             # At 10:15, the 9:30 bar is effectively "Current". It is not closed.
             # The previous closed bar is... yesterday's close? Or premarket?
             
             # Wait. If we are training, we want "Predict next 15m Close".
             # At 10:15 (predicting 10:30), we know:
             # - 15m bar 10:00-10:15 (Just closed).
             # - 1H bar? The 9:30-10:30 bar is NOT closed.
             # - The 8:30-9:30 bar IS closed.
             
             # So we need data from 2 hours ago? or 1 hour ago?
             # 1H Bar 9:30 contains data up to 10:30.
             # We cannot use ANY of it at 10:15.
             # So we must use the row BEFORE 9:30.
             
             # Current code `fetch_and_process(shift=1)`:
             # Row 9:30 contains Row 8:30 data.
             # At 10:15, merge_asof backwards finds 9:30.
             # It sees "Row 8:30 data". 
             # Is 8:30 data safe? Yes, it closed at 9:30.
             # So referencing 9:30 label (which holds 8:30 data) is safe.
             
             # Limitation: We lose the "Live" intra-bar development of the 1h bar (e.g. current 1H open).
             # That's acceptable for "Structure".
             
             merged = pd.merge_asof(
                 base_df, 
                 df_1h, 
                 left_index=True, 
                 right_index=True, 
                 direction='backward',
                 suffixes=('', '_1h')
             )
             base_df = merged

        # Fill NaNs (from lags)
        base_df = base_df.ffill().dropna()
        
        print(f"Data Prepared. Shape: {base_df.shape}")
        return base_df

