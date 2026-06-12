"""
Stock Predictor Engine
Implements machine learning models for forecasting stock price movements.
Uses LSTM (deep learning) and XGBoost (gradient boosting) with ensemble approach.

Supports:
- Price/return prediction (regression)
- Direction prediction (classification: up/down)
- Model evaluation with MAE, RMSE, R², and Directional Accuracy
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import warnings
warnings.filterwarnings("ignore")

from config import (
    TRAIN_TEST_SPLIT_RATIO, VALIDATION_SPLIT, RANDOM_SEED,
    PREDICTION_HORIZON, MODELS_DIR,
    XGBOOST_N_ESTIMATORS, XGBOOST_MAX_DEPTH, XGBOOST_LEARNING_RATE,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE,
)

np.random.seed(RANDOM_SEED)


# ============================================================
# Evaluation Metrics
# ============================================================

def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy - the proportion of times
    the model correctly predicts the direction of movement.
    """
    y_true_dir = np.sign(y_true)
    y_pred_dir = np.sign(y_pred)
    return np.mean(y_true_dir == y_pred_dir)


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Compute and return all regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dir_acc = directional_accuracy(y_true, y_pred)

    results = {
        'Model': model_name,
        'MAE': round(mae, 6),
        'RMSE': round(rmse, 6),
        'R2_Score': round(r2, 4),
        'Directional_Accuracy': round(dir_acc * 100, 2),
    }
    return results


# ============================================================
# XGBoost Predictor
# ============================================================

class XGBoostPredictor:
    """
    XGBoost-based stock predictor.
    Works well with tabular feature data and provides good interpretability
    through feature importance scores.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def train(self, X_train, y_train, feature_names=None):
        """Train the XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            print("  XGBoost not installed. Install with: pip install xgboost")
            return None

        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=XGBOOST_N_ESTIMATORS,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        self.model.fit(
            X_scaled, y_train,
            eval_set=[(X_scaled, y_train)],
            verbose=False,
        )

        self.is_trained = True
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        """Get feature importance scores."""
        if not self.is_trained:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def get_top_features(self, n=15):
        """Get the top N most important features."""
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def save(self, filepath):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
            }, f)

    def load(self, filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_trained = True
        return self


# ============================================================
# LSTM Predictor (Deep Learning)
# ============================================================

class LSTMPredictor:
    """
    LSTM (Long Short-Term Memory) neural network for time series forecasting.
    Good at capturing sequential patterns in stock price data.
    """

    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.training_history = None

    def _create_sequences(self, data):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train(self, prices, epochs=None, batch_size=None):
        """
        Train LSTM on price data.
        
        Parameters:
        -----------
        prices : array-like, price series (typically 'close' prices)
        epochs : int, number of training epochs
        batch_size : int, batch size for training
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            print("  TensorFlow/Keras not installed. Install with: pip install tensorflow")
            return None

        epochs = epochs or LSTM_EPOCHS
        batch_size = batch_size or LSTM_BATCH_SIZE

        # Scale data
        prices = np.array(prices).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)

        # Create sequences
        X, y = self._create_sequences(scaled_data.flatten())

        if len(X) == 0:
            print("  Not enough data to create sequences")
            return None

        # Reshape for LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Train/validation split
        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build LSTM model
        self.model = Sequential([
            LSTM(LSTM_UNITS[0], return_sequences=True,
                 input_shape=(self.lookback, 1)),
            Dropout(LSTM_DROPOUT),
            LSTM(LSTM_UNITS[1], return_sequences=False),
            Dropout(LSTM_DROPOUT),
            Dense(32, activation='relu'),
            Dense(1),
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=0
        )

        # Train
        self.training_history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0,
        )

        self.is_trained = True
        return self

    def predict_future(self, prices, days_ahead=30):
        """
        Predict future prices for a given number of days.
        Uses iterative prediction (each prediction feeds into the next).
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet.")

        prices = np.array(prices).reshape(-1, 1)
        scaled = self.scaler.transform(prices).flatten()

        predictions = []
        current_sequence = scaled[-self.lookback:].copy()

        for _ in range(days_ahead):
            seq = current_sequence[-self.lookback:].reshape(1, self.lookback, 1)
            pred = self.model.predict(seq, verbose=0)[0, 0]
            predictions.append(pred)
            current_sequence = np.append(current_sequence, pred)

        # Inverse transform to get actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()

        return predictions

    def save(self, filepath):
        """Save model to disk."""
        if self.model is None:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save Keras model
        keras_path = filepath.replace('.pkl', '.keras')
        self.model.save(keras_path)

        # Save scaler separately
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'lookback': self.lookback,
            }, f)

    def load(self, filepath):
        """Load model from disk."""
        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            print("  TensorFlow not installed")
            return None

        keras_path = filepath.replace('.pkl', '.keras')
        self.model = load_model(keras_path)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.lookback = data['lookback']
        self.is_trained = True
        return self


# ============================================================
# Ensemble Predictor
# ============================================================

class EnsemblePredictor:
    """
    Combines multiple models for more robust predictions.
    Uses weighted averaging of XGBoost and LSTM predictions.
    """

    def __init__(self):
        self.xgboost_model = XGBoostPredictor()
        self.lstm_model = LSTMPredictor()
        self.xgb_weight = 0.5
        self.lstm_weight = 0.5
        self.is_trained = False

    def train(self, X_train, y_train, prices, feature_names=None):
        """Train both models."""
        print("  Training XGBoost model...")
        self.xgboost_model.train(X_train, y_train, feature_names)

        print("  Training LSTM model...")
        self.lstm_model.train(prices)

        self.is_trained = True
        return self

    def predict(self, X, prices=None):
        """Get ensemble prediction."""
        if not self.is_trained:
            raise ValueError("Models not trained yet.")

        predictions = []

        if self.xgboost_model.is_trained:
            xgb_pred = self.xgboost_model.predict(X)
            predictions.append(xgb_pred * self.xgb_weight)

        if self.lstm_model.is_trained and prices is not None:
            lstm_pred = self.lstm_model.predict_future(prices, days_ahead=len(X))
            predictions.append(lstm_pred * self.lstm_weight)

        if not predictions:
            return np.zeros(len(X))

        # Weighted ensemble
        return np.sum(predictions, axis=0) / sum([self.xgb_weight, self.lstm_weight])


# ============================================================
# Stock Prediction Engine (Main Class)
# ============================================================

class StockPredictionEngine:
    """
    Main engine that handles the full prediction pipeline:
    feature preparation -> model training -> evaluation -> forecasting.
    """

    def __init__(self):
        self.xgboost_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.evaluation_results = {}
        self.is_trained = False

    def train_for_stock(self, df_with_features, stock_symbol, target_horizon=PREDICTION_HORIZON):
        """
        Train prediction models for a specific stock.
        
        Parameters:
        -----------
        df_with_features : DataFrame with all technical indicators computed
        stock_symbol : str, name of the stock
        target_horizon : int, days ahead to predict
        
        Returns:
        --------
        dict with evaluation results
        """
        from feature_engineering import FeatureEngineer

        print(f"\n  Training models for {stock_symbol}...")
        print(f"  Target horizon: {target_horizon} days")

        df = df_with_features.copy()

        # Prepare target
        df['target_return'] = df['close'].shift(-target_horizon) / df['close'] - 1
        df['target_direction'] = (df['target_return'] > 0).astype(int)

        # Select feature columns
        exclude = [
            'date', 'symbol', 'series', 'target_return', 'target_direction',
            'daily_return', 'log_return', 'cumulative_return', 'price_direction',
        ]

        feature_cols = [c for c in df.columns
                        if c not in exclude
                        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        self.feature_names = feature_cols

        # Drop NaN rows
        df_clean = df.dropna(subset=feature_cols + ['target_return'])

        if len(df_clean) < 100:
            print(f"  Not enough clean data for {stock_symbol} (only {len(df_clean)} rows)")
            return None

        print(f"  Clean data points: {len(df_clean)}")
        print(f"  Features: {len(feature_cols)}")

        # Prepare X and y
        X = df_clean[feature_cols].values
        y_return = df_clean['target_return'].values

        # Train/test split (time-series aware - no shuffling)
        split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_return[:split_idx], y_return[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ---- Train XGBoost ----
        print("  [1/2] Training XGBoost...")
        try:
            import xgboost as xgb
            self.xgboost_model = xgb.XGBRegressor(
                n_estimators=XGBOOST_N_ESTIMATORS,
                max_depth=XGBOOST_MAX_DEPTH,
                learning_rate=XGBOOST_LEARNING_RATE,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
                verbosity=0,
            )
            self.xgboost_model.fit(X_train_scaled, y_train, verbose=False)
            xgb_pred = self.xgboost_model.predict(X_test_scaled)
        except ImportError:
            print("  XGBoost not available, skipping...")
            self.xgboost_model = None
            xgb_pred = None

        # ---- Train LSTM ----
        print("  [2/2] Training LSTM...")
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.optimizers import Adam

            close_prices = df_clean['close'].values
            price_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = price_scaler.fit_transform(close_prices.reshape(-1, 1))

            # Create sequences
            lookback = 60
            X_lstm, y_lstm = [], []
            for i in range(lookback, len(scaled_prices)):
                X_lstm.append(scaled_prices[i-lookback:i])
                y_lstm.append(scaled_prices[i])
            X_lstm = np.array(X_lstm).reshape(-1, lookback, 1)
            y_lstm = np.array(y_lstm)

            lstm_split = int(len(X_lstm) * TRAIN_TEST_SPLIT_RATIO)
            X_lstm_train, X_lstm_test = X_lstm[:lstm_split], X_lstm[lstm_split:]
            y_lstm_train, y_lstm_test = y_lstm[:lstm_split], y_lstm[lstm_split:]

            self.lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1),
            ])
            self.lstm_model.compile(optimizer=Adam(0.001), loss='mse')

            early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                       restore_best_weights=True, verbose=0)

            self.lstm_model.fit(
                X_lstm_train, y_lstm_train,
                epochs=50, batch_size=32,
                validation_data=(X_lstm_test, y_lstm_test),
                callbacks=[early_stop], verbose=0,
            )
        except ImportError:
            print("  TensorFlow/Keras not available, skipping LSTM...")
            self.lstm_model = None

        # ---- Evaluate XGBoost ----
        results = {}
        if xgb_pred is not None:
            results['XGBoost'] = evaluate_regression(y_test, xgb_pred, "XGBoost")
            print(f"\n  XGBoost Results:")
            for k, v in results['XGBoost'].items():
                if k != 'Model':
                    print(f"    {k}: {v}")

        self.evaluation_results = results
        self.is_trained = True

        return results

    def predict_future(self, df_with_features, days_ahead=30):
        """
        Generate future predictions for a stock.
        Uses XGBoost for feature-based prediction.
        """
        if not self.is_trained or self.xgboost_model is None:
            print("  Model not trained. Train first.")
            return None

        df = df_with_features.copy()

        exclude = [
            'date', 'symbol', 'series',
            'daily_return', 'log_return', 'cumulative_return', 'price_direction',
        ]
        feature_cols = [c for c in df.columns if c in self.feature_names]

        if not feature_cols:
            print("  No matching features found")
            return None

        X = df[feature_cols].iloc[-1:].values
        X_scaled = self.scaler.transform(X)

        predicted_return = self.xgboost_model.predict(X_scaled)[0]
        current_price = df['close'].iloc[-1]

        predicted_price = current_price * (1 + predicted_return)

        # Also try LSTM if available
        lstm_prediction = None
        if self.lstm_model is not None:
            try:
                close_prices = df['close'].values
                price_scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = price_scaler.fit_transform(close_prices.reshape(-1, 1))

                lookback = 60
                last_seq = scaled[-lookback:].reshape(1, lookback, 1)
                lstm_scaled_pred = self.lstm_model.predict(last_seq, verbose=0)[0, 0]
                lstm_prediction = price_scaler.inverse_transform([[lstm_scaled_pred]])[0, 0]
            except Exception:
                pass

        result = {
            'current_price': current_price,
            'predicted_return_pct': round(predicted_return * 100, 2),
            'predicted_price_xgboost': round(predicted_price, 2),
            'predicted_price_lstm': round(lstm_prediction, 2) if lstm_prediction else None,
            'direction': 'UP' if predicted_return > 0 else 'DOWN',
            'horizon_days': days_ahead,
        }

        return result

    def get_model_interpretation(self):
        """Get feature importance from XGBoost for explainability."""
        if self.xgboost_model is None:
            return []

        try:
            importance = self.xgboost_model.feature_importances_
            feature_imp = list(zip(self.feature_names, importance))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            return feature_imp[:20]
        except Exception:
            return []

    def save_models(self, stock_symbol):
        """Save trained models to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)

        if self.xgboost_model is not None:
            path = os.path.join(MODELS_DIR, f"{stock_symbol}_xgb.pkl")
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.xgboost_model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                }, f)

        if self.lstm_model is not None:
            path = os.path.join(MODELS_DIR, f"{stock_symbol}_lstm.keras")
            self.lstm_model.save(path)

        print(f"  Models saved for {stock_symbol}")
