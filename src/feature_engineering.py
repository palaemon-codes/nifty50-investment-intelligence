"""
Feature Engineering Module
Computes technical indicators and derived features from raw stock price data.
Includes Moving Averages, RSI, MACD, Bollinger Bands, and momentum indicators.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    MA_PERIODS, EMA_PERIODS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD,
    ATR_PERIOD, STOCHASTIC_K_PERIOD, STOCHASTIC_D_PERIOD,
)


class FeatureEngineer:
    """
    Adds technical indicators and derived features to stock price DataFrames.
    These features are used by the prediction models and for analysis.
    """

    def __init__(self):
        self.added_features = []

    def compute_all_features(self, df):
        """
        Compute all technical indicators and add them to the DataFrame.
        Returns the enriched DataFrame.
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        # Make sure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                print(f"  Warning: Column '{col}' not found, skipping some features")
                return df

        # ---- Price-based features ----
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_exponential_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)

        # ---- Volume-based features ----
        df = self.add_volume_features(df)

        # ---- Momentum features ----
        df = self.add_momentum_features(df)
        df = self.add_stochastic_oscillator(df)

        # ---- Volatility features ----
        df = self.add_volatility_features(df)

        # ---- Price pattern features ----
        df = self.add_price_position_features(df)

        # Record what we added
        self.added_features = [c for c in df.columns if c not in
                               ['date', 'open', 'high', 'low', 'close', 'volume',
                                'turnover', 'symbol', 'series', 'vwap', 'adj_close']]

        return df

    # ----------------------------------------------------------
    # Individual feature methods
    # ----------------------------------------------------------

    def add_returns(self, df):
        """Calculate daily returns and log returns."""
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        # Cumulative returns from start
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        return df

    def add_moving_averages(self, df):
        """Add Simple Moving Averages (SMA) for multiple periods."""
        for period in MA_PERIODS:
            col_name = f'sma_{period}'
            df[col_name] = df['close'].rolling(window=period).mean()
            # Price relative to MA (useful signal)
            df[f'price_to_sma_{period}'] = df['close'] / df[col_name] - 1
        return df

    def add_exponential_moving_averages(self, df):
        """Add Exponential Moving Averages (EMA)."""
        for period in EMA_PERIODS:
            col_name = f'ema_{period}'
            df[col_name] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def add_rsi(self, df):
        """Add Relative Strength Index (RSI)."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=RSI_PERIOD).mean()

        # Use Wilder's smoothing for more accuracy
        for i in range(RSI_PERIOD, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (RSI_PERIOD - 1) + gain.iloc[i]) / RSI_PERIOD
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (RSI_PERIOD - 1) + loss.iloc[i]) / RSI_PERIOD

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI signal: overbought > 70, oversold < 30
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

        return df

    def add_macd(self, df):
        """Add MACD (Moving Average Convergence Divergence)."""
        ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD crossover signals
        df['macd_bullish_cross'] = (
            (df['macd'] > df['macd_signal']) &
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)

        df['macd_bearish_cross'] = (
            (df['macd'] < df['macd_signal']) &
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)

        return df

    def add_bollinger_bands(self, df):
        """Add Bollinger Bands."""
        rolling_mean = df['close'].rolling(window=BOLLINGER_PERIOD).mean()
        rolling_std = df['close'].rolling(window=BOLLINGER_PERIOD).std()

        df['bb_middle'] = rolling_mean
        df['bb_upper'] = rolling_mean + (BOLLINGER_STD * rolling_std)
        df['bb_lower'] = rolling_mean - (BOLLINGER_STD * rolling_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        # %B indicator
        df['bb_percent_b'] = df['bb_position']

        return df

    def add_atr(self, df):
        """Add Average True Range (ATR) - volatility indicator."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=ATR_PERIOD).mean()
        # Normalized ATR
        df['atr_percent'] = df['atr'] / df['close'] * 100

        return df

    def add_volume_features(self, df):
        """Add volume-based features."""
        # Volume moving average
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_50'] = df['volume'].rolling(window=50).mean()

        # Volume ratio (current volume / average volume)
        df['volume_ratio'] = df['volume'] / df['volume_sma_50']

        # Volume trend
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_50'] - 1

        # On-Balance Volume (OBV)
        df['price_direction'] = np.where(df['close'] > df['close'].shift(1), 1,
                                         np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['obv'] = (df['volume'] * df['price_direction']).cumsum()

        # Volume-price trend
        df['vpt'] = (df['volume'] * df['daily_return']).cumsum()

        return df

    def add_momentum_features(self, df):
        """Add momentum-based indicators."""
        # Price rate of change (ROC) for different periods
        for period in [5, 10, 20, 60]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100

        # Momentum: close - close N days ago
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)

        return df

    def add_stochastic_oscillator(self, df):
        """Add Stochastic Oscillator (%K and %D)."""
        low_min = df['low'].rolling(window=STOCHASTIC_K_PERIOD).min()
        high_max = df['high'].rolling(window=STOCHASTIC_K_PERIOD).max()

        df['stoch_k'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=STOCHASTIC_D_PERIOD).mean()

        return df

    def add_volatility_features(self, df):
        """Add volatility-based features."""
        # Historical volatility (rolling standard deviation of returns)
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}d'] = df['daily_return'].rolling(window=period).std()

        # Annualized 20-day volatility
        df['volatility_annualized'] = df['volatility_20d'] * np.sqrt(252)

        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
        )
        df['parkinson_vol_20d'] = df['parkinson_vol'].rolling(window=20).mean()

        return df

    def add_price_position_features(self, df):
        """Add features describing the price position relative to ranges."""
        # Where is close relative to day's range?
        df['price_position_daily'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # 52-week (252 trading days) high and low
        df['high_52w'] = df['high'].rolling(window=252).max()
        df['low_52w'] = df['low'].rolling(window=252).min()
        df['price_vs_52w_high'] = df['close'] / df['high_52w'] - 1
        df['price_vs_52w_low'] = df['close'] / df['low_52w'] - 1

        # Distance from 200-day MA (often used as trend indicator)
        if 'sma_200' in df.columns:
            df['distance_from_200ma'] = df['close'] / df['sma_200'] - 1

        return df

    def get_feature_columns(self):
        """Return list of all feature column names added."""
        return self.added_features

    def prepare_ml_dataset(self, df, target_horizon=30):
        """
        Prepare a dataset suitable for machine learning.
        
        Parameters:
        -----------
        df : DataFrame with computed features
        target_horizon : int, number of days ahead to predict
        
        Returns:
        --------
        X : Feature matrix
        y : Target vector (future return over horizon)
        feature_names : List of feature column names
        """
        df = df.copy()

        # Target: future return over the horizon
        df['target_future_return'] = df['close'].shift(-target_horizon) / df['close'] - 1
        df['target_direction'] = (df['target_future_return'] > 0).astype(int)

        # Select feature columns (exclude non-feature columns)
        exclude_cols = [
            'date', 'symbol', 'series', 'target_future_return', 'target_direction',
            'daily_return', 'log_return', 'cumulative_return',
        ]

        feature_cols = [c for c in df.columns if c not in exclude_cols
                        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        # Drop rows with NaN values
        df_clean = df.dropna(subset=feature_cols + ['target_future_return'])

        X = df_clean[feature_cols].values
        y_return = df_clean['target_future_return'].values
        y_direction = df_clean['target_direction'].values

        return X, y_return, y_direction, feature_cols, df_clean
