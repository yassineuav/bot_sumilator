import numpy as np
import pandas as pd
# import tensorflow as tf moved down
# from tensorflow.keras.models import Sequential, load_model moved down
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input moved down
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime

import os

# Disable oneDNN and GPU BEFORE importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Disable GPU if not needed or causing issues (optional, depends on env)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trained_models', 'lstm')

class LSTMModel:
    def __init__(self, symbol, interval, lookback=60, project_name='default'):
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.project_name = project_name
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Base dir for this symbol/interval
        self.base_dir = os.path.join(MODEL_BASE_DIR, project_name, symbol, interval)
        self.versions_dir = os.path.join(self.base_dir, 'versions')
        
        # Ensure dirs exist
        if not os.path.exists(self.versions_dir):
            os.makedirs(self.versions_dir)
            
    def _create_sequences(self, data, target_col_idx):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback, target_col_idx])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1)) # Predicting Close price
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self, df, epochs=20, batch_size=32, target_col='Close'):
        # 1. Preprocess
        # Filter for numerical columns only, exclude non-feature columns
        exclude_cols = ['Datetime', 'Date', 'symbol', 'target', 'signal', 'confidence', 'future_close', 'label', 'future_return', 'pnl', 'pnl_pct', 'exit_reason']
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        # Ensure Target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")
            
        data = df[feature_cols].values
        self.feature_cols = feature_cols # Save for prediction
        
        # Scale
        scaled_data = self.scaler.fit_transform(data)
        
        # Target column index (for creating y)
        if target_col in feature_cols:
            target_idx = feature_cols.index(target_col)
        else:
             # If target is not in features (e.g. future prediction target), handle differently?
             # For now assume auto-regression on Close
             target_idx = feature_cols.index(target_col)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, target_idx)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
            
        # Train
        print(f"Training LSTM for {self.symbol} {self.interval}...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Eval
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Final Loss: {loss}")
        
        self.save_model(history.history)
        return history.history

    def _get_active_version_path(self):
        # path to file that stores which version is active
        pointer_path = os.path.join(self.base_dir, 'active_version.json')
        if os.path.exists(pointer_path):
            with open(pointer_path, 'r') as f:
                data = json.load(f)
                return data.get('version_timestamp')
        return None

    def set_active_version(self, version_timestamp):
        # Check if version exists
        version_path = os.path.join(self.versions_dir, version_timestamp)
        if not os.path.exists(version_path):
            raise ValueError(f"Version {version_timestamp} does not exist")
            
        pointer_path = os.path.join(self.base_dir, 'active_version.json')
        with open(pointer_path, 'w') as f:
            json.dump({'version_timestamp': version_timestamp}, f)
        print(f"Active version set to {version_timestamp}")

    def list_versions(self):
        # Return list of versions with metadata
        versions = []
        if not os.path.exists(self.versions_dir):
            return []
            
        active_ts = self._get_active_version_path()
        
        for ts in os.listdir(self.versions_dir):
            meta_path = os.path.join(self.versions_dir, ts, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    try:
                        meta = json.load(f)
                        meta['is_active'] = (ts == active_ts)
                        meta['version_id'] = ts
                        versions.append(meta)
                    except:
                        pass
                        
        # Sort by date desc
        versions.sort(key=lambda x: x.get('last_trained', ''), reverse=True)
        return versions

    def save_model(self, history=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_path = os.path.join(self.versions_dir, timestamp)
        os.makedirs(version_path)
        
        # Save Keras Model
        model_path = os.path.join(version_path, 'model.keras')
        self.model.save(model_path)
        
        # Save Scaler
        scaler_path = os.path.join(version_path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save Metadata
        meta_path = os.path.join(version_path, 'metadata.json')
        metadata = {
            'symbol': self.symbol,
            'interval': self.interval,
            'lookback': self.lookback,
            'last_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'history': history if history else {},
            # Summary metrics
            'final_loss': history['loss'][-1] if history and 'loss' in history else None,
            'feature_cols': getattr(self, 'feature_cols', [])
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # Set as active automatically
        self.set_active_version(timestamp)
        print(f"LSTM Model saved to {version_path}")

    def load(self, version_timestamp=None):
        if version_timestamp is None:
            version_timestamp = self._get_active_version_path()
            
        if not version_timestamp:
            print("No active model version found.")
            return False
            
        version_path = os.path.join(self.versions_dir, version_timestamp)
        model_path = os.path.join(version_path, 'model.keras')
        scaler_path = os.path.join(version_path, 'scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            # Load metadata for feature_cols
            meta_path = os.path.join(version_path, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    self.feature_cols = meta.get('feature_cols', ['Open', 'High', 'Low', 'Close', 'Volume'])
            else:
                 self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                 
            print(f"Loaded LSTM model version {version_timestamp} for {self.symbol} {self.interval}")
            return True
        return False

    def predict_next(self, df):
        # Predict next Close price
        # df should have at least 'lookback' rows
        feature_cols = getattr(self, 'feature_cols', ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Check if all cols exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
             # If exact cols missing, maybe try to be flexible? 
             # No, structure must match.
             raise ValueError(f"Missing columns for prediction: {missing}")

        recent_data = df[feature_cols].tail(self.lookback).values
        if len(recent_data) < self.lookback:
            raise ValueError(f"Not enough data to predict. Need {self.lookback} rows.")
            
        # Scale
        scaled_data = self.scaler.transform(recent_data)
        
        # Reshape to (1, lookback, features)
        X_input = scaled_data.reshape(1, self.lookback, len(feature_cols))
        
        # Predict
        predicted_scaled = self.model.predict(X_input)
        
        # Inverse logic
        # We need to map the predicted single value back.
        # Implemented for target_idx = Close
        if 'Close' in feature_cols:
             target_idx = feature_cols.index('Close')
        else:
             target_idx = 3 # Fallback default
             
        min_val = self.scaler.data_min_[target_idx]
        max_val = self.scaler.data_max_[target_idx]
        
        # X = X_std * (max - min) + min
        predicted_price = predicted_scaled[0, 0] * (max_val - min_val) + min_val
        
        return predicted_price

    def predict_signals(self, df):
        """
        Generate predictions for the entire dataframe (where possible) for backtesting.
        Returns a DataFrame with 'lstm_pred' column and 'signal' column.
        """
        feature_cols = getattr(self, 'feature_cols', ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Validation
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
             print(f"Warning: Missing columns {missing}. prediction skipped.")
             return df
             
        data = df[feature_cols].values # (N, Features)
        
        # Scale entire dataset
        # Scale entire dataset
        scaled_data = self.scaler.transform(data)
        
        if 'Close' in feature_cols:
             target_idx = feature_cols.index('Close')
        else:
             target_idx = 3 # Fallback
        
        # Create sequences for entire history
        # We can predict from index `lookback` onwards
        X_all = []
        valid_indices = []
        
        for i in range(len(scaled_data) - self.lookback):
            X_all.append(scaled_data[i:(i + self.lookback)])
            valid_indices.append(df.index[i + self.lookback])
            
        X_all = np.array(X_all)
        
        if len(X_all) == 0:
            return df # No predictions possible
            
        # Batch predict
        preds_scaled = self.model.predict(X_all, verbose=0)
        
        # Inverse transform
        min_val = self.scaler.data_min_[target_idx]
        max_val = self.scaler.data_max_[target_idx]
        preds = preds_scaled.flatten() * (max_val - min_val) + min_val
        
        # Align with original DF
        # The prediction at index `lookback` corresponds to the price at `lookback`?
        # NO. The model is trained to predict `lookback` index based on `0..lookback-1`.
        # So X[i] (which is data[i:i+LB]) predicts y[i] (which is data[i+LB]).
        # So prediction `preds[0]` corresponds to time `valid_indices[0]`.
        
        # Create results Series
        pred_series = pd.Series(preds, index=valid_indices)
        
        # Merge into DF
        df['lstm_pred'] = pred_series
        
        # Generate Signals
        # Logic: If Predicted Next > Current Close -> CALL
        # Wait, `lstm_pred` at time T is the prediction for time T made at T-1?
        # Or is it the prediction made at T for T+1?
        # In training: X (0..59) -> y (60).
        # So at time 59 (end of X), we predict Price(60).
        # So let's shift the prediction.
        # The calculation `preds[0]` used data up to `index[lookback-1]`.
        # It is a prediction FOR `index[lookback]`.
        # So `lstm_pred` at row T is "What the model predicted Price(T) would be".
        
        # Signal Generation:
        # If at time T, the model predicts T+1 > Close(T), then Buy at T.
        # We need "Future Prediction" column.
        
        # Let's align it carefully.
        # We want: df['future_pred'] where row T contains prediction for T+1.
        # prediction made using data ending at T is `model.predict(data[T-LB+1 : T+1])`.
        # The loop above: `X_all[k]` uses data `k` to `k+LB`.
        # It predicts `data[k+LB]`.
        # So at row `k+LB-1` (the last data point used in input), we have this prediction.
        
        # Let's re-index.
        # Prediction `preds[k]` is derived from data ending at `valid_indices[k] - 1 step`.
        # So we should assign `preds[k]` to the row BEFORE `valid_indices[k]`.
        
        # Shifted index:
        shifted_indices = df.index[self.lookback-1 : -1] 
        # Length check
        if len(shifted_indices) != len(preds):
            # If we predict for the very last step (future unknown), we might have 1 extra prediction?
            # In X generation above: range(len - lookback).
            # Last i is len - lookback - 1.
            # Window is [len-lookback-1 : len-1]. (Last available full window).
            # Predicts index [len-1].
            # So `preds` contains predictions for indices [lookback ... len-1].
            # These are predictions FOR those timestamps.
            # We want to know what the prediction WAS at timestamp T-1.
            pass

        df['lstm_price_at_close'] = pred_series # aligned to when the price actually happened
        
        # Signal:
        # At row T, we want to know: Did we predict Price(T+1) > Price(T)?
        # That prediction comes from row T.
        # Prediction for T+1 is `lstm_price_at_close` shifted backwards by 1?
        # No. `lstm_price_at_close` at T+1 is the prediction FOR T+1.
        # It was available at T.
        
        df['next_price_pred'] = df['lstm_price_at_close'].shift(-1)
        
        # If Next Price Pred > Current Close -> LONG
        # If Next Price Pred < Current Close -> SHORT
        
        conditions = [
            (df['next_price_pred'] > df['Close']),
            (df['next_price_pred'] < df['Close'])
        ]
        choices = ['CALL', 'PUT']
        df['signal'] = np.select(conditions, choices, default='NEUTRAL')
        
        # Confidence? We are doing regression (price), not classification (prob).
        # Maybe use % diff as confidence proxy?
        df['confidence'] = abs(df['next_price_pred'] - df['Close']) / df['Close']
        
        return df
