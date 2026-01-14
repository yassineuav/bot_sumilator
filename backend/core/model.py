import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

MODEL_PATH = 'trading_model.joblib'

class TradingModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            objective='multi:softmax',
            num_class=3, # -1, 0, 1 mapped to 0, 1, 2
            random_state=42
        )
        self.feature_cols = None

    def _prepare_data(self, df):
        # Map labels: -1 -> 0, 0 -> 1, 1 -> 2
        y = df['target'].map({-1: 0, 0: 1, 1: 2})
        X = df.drop(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'target', 'label', 'future_close', 'future_return', 'Capital Gains', 'Dividends', 'Stock Splits'], errors='ignore')
        self.feature_cols = X.columns.tolist()
        return X, y

    def train(self, df):
        X, y = self._prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"Training model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("Model Evaluation:")
        print(classification_report(y_test, y_pred, target_names=['Bearish', 'Neutral', 'Bullish']))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        
        self.save()

    def predict(self, df):
        # In case we drop rows during feature engineering, we want to return the full df with predictions
        X = df[self.feature_cols]
        preds = self.model.predict(X)
        # Map back: 0 -> -1, 1 -> 0, 2 -> 1
        return pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

    def predict_proba(self, df):
        X = df[self.feature_cols]
        return self.model.predict_proba(X)

    def save(self, path=MODEL_PATH):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols}, path)
        print(f"Model saved to {path}")

    def load(self, path=MODEL_PATH):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.feature_cols = data['feature_cols']
            print(f"Model loaded from {path}")
            return True
        return False

if __name__ == "__main__":
    import data_loader
    import features
    import patterns
    
    # 1. Get Data
    df = data_loader.fetch_data('SPY', interval='15m', period='60d')
    # 2. Add Features
    df = features.compute_features(df)
    # 3. Label Data
    df = patterns.label_data(df)
    
    # 4. Train Model
    tm = TradingModel()
    tm.train(df)
