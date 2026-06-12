"""
Market Anomaly Detection Module
Identifies unusual patterns and events in historical market data.

Detects:
- Sudden volatility spikes
- Extreme price movements (flash crashes / rallies)
- Unusual trading volume surges
- Regime changes in market behavior
- Structural breaks in price trends
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import (
    ANOMALY_ZSCORE_THRESHOLD, ANOMALY_VOLUME_SURGE_MULTIPLIER,
    ANOMALY_LOOKBACK_DAYS,
)


class AnomalyDetector:
    """
    Detects market anomalies using statistical methods.
    Works with both individual stocks and market-wide analysis.
    """

    def __init__(self, zscore_threshold=ANOMALY_ZSCORE_THRESHOLD):
        self.zscore_threshold = zscore_threshold
        self.detected_anomalies = {}

    # ================================================================
    # Price Anomaly Detection
    # ================================================================

    def detect_price_anomalies(self, df, symbol=None):
        """
        Detect anomalous price movements using statistical methods.
        
        Methods used:
        1. Z-score on daily returns
        2. Interquartile Range (IQR) method
        3. Modified Z-score (uses median instead of mean)
        """
        if df is None or len(df) < 50:
            return pd.DataFrame()

        df = df.copy()
        returns = df['close'].pct_change()

        # Method 1: Z-score on returns
        returns_mean = returns.mean()
        returns_std = returns.std()
        z_scores = np.abs((returns - returns_mean) / returns_std)
        df['anomaly_zscore'] = z_scores > self.zscore_threshold

        # Method 2: IQR method (robust to outliers)
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['anomaly_iqr'] = (returns < lower_bound) | (returns > upper_bound)

        # Method 3: Modified Z-score (median-based, more robust)
        returns_median = returns.median()
        mad = np.median(np.abs(returns - returns_median))  # Median Absolute Deviation
        if mad > 0:
            modified_z = 0.6745 * (returns - returns_median) / mad
            df['anomaly_modified_z'] = np.abs(modified_z) > 3.5
        else:
            df['anomaly_modified_z'] = False

        # Combined anomaly flag (at least 2 out of 3 methods agree)
        anomaly_flags = df[['anomaly_zscore', 'anomaly_iqr', 'anomaly_modified_z']].sum(axis=1)
        df['is_anomaly'] = anomaly_flags >= 2

        # Mark anomaly type
        df['anomaly_type'] = 'normal'
        df.loc[df['is_anomaly'] & (returns > 0), 'anomaly_type'] = 'extreme_up'
        df.loc[df['is_anomaly'] & (returns < 0), 'anomaly_type'] = 'extreme_down'

        # Extract anomaly dates
        anomalies = df[df['is_anomaly']].copy()
        anomalies['symbol'] = symbol or 'Unknown'

        # Store results
        if symbol:
            self.detected_anomalies[symbol] = {
                'total_anomalies': len(anomalies),
                'anomaly_pct': round(len(anomalies) / len(df) * 100, 2),
                'extreme_up': len(anomalies[anomalies['anomaly_type'] == 'extreme_up']),
                'extreme_down': len(anomalies[anomalies['anomaly_type'] == 'extreme_down']),
                'anomaly_df': anomalies,
            }

        return anomalies

    # ================================================================
    # Volume Anomaly Detection
    # ================================================================

    def detect_volume_anomalies(self, df, symbol=None):
        """
        Detect unusual trading volume activity.
        Flags days where volume is significantly above normal levels.
        """
        if df is None or len(df) < 50:
            return pd.DataFrame()

        df = df.copy()

        # Volume moving average and standard deviation
        volume_mean = df['volume'].rolling(window=50).mean()
        volume_std = df['volume'].rolling(window=50).std()

        # Volume Z-score
        df['volume_zscore'] = (df['volume'] - volume_mean) / volume_std
        df['volume_anomaly'] = df['volume_zscore'] > ANOMALY_VOLUME_SURGE_MULTIPLIER

        # Volume surge ratio (current volume vs 50-day average)
        df['volume_surge_ratio'] = df['volume'] / volume_mean
        df['volume_surge'] = df['volume_surge_ratio'] > ANOMALY_VOLUME_SURGE_MULTIPLIER

        # Combined volume anomaly
        df['is_volume_anomaly'] = df['volume_anomaly'] | df['volume_surge']

        volume_anomalies = df[df['is_volume_anomaly']].copy()
        volume_anomalies['symbol'] = symbol or 'Unknown'

        return volume_anomalies

    # ================================================================
    # Volatility Spike Detection
    # ================================================================

    def detect_volatility_spikes(self, df, symbol=None):
        """
        Detect sudden increases in volatility.
        Identifies periods where volatility jumps significantly above
        its recent historical average.
        """
        if df is None or len(df) < 60:
            return pd.DataFrame()

        df = df.copy()

        # Calculate rolling volatility (20-day, annualized)
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        vol_mean = rolling_vol.rolling(window=ANOMALY_LOOKBACK_DAYS).mean()
        vol_std = rolling_vol.rolling(window=ANOMALY_LOOKBACK_DAYS).std()

        # Volatility Z-score
        df['volatility'] = rolling_vol
        df['volatility_mean'] = vol_mean
        df['volatility_zscore'] = (rolling_vol - vol_mean) / vol_std
        df['volatility_spike'] = df['volatility_zscore'] > self.zscore_threshold

        spikes = df[df['volatility_spike']].copy()
        spikes['symbol'] = symbol or 'Unknown'

        return spikes

    # ================================================================
    # Regime Change Detection
    # ================================================================

    def detect_regime_changes(self, df, symbol=None):
        """
        Detect structural breaks or regime changes in price behavior.
        Uses rolling statistics to identify shifts in market regime.
        """
        if df is None or len(df) < 252:
            return []

        df = df.copy()
        returns = df['close'].pct_change()  # Keep NaN at position 0 so lengths align with df

        # Calculate rolling statistics
        roll_mean_60d = returns.rolling(window=60).mean() * 252  # Annualized
        roll_vol_60d = returns.rolling(window=60).std() * np.sqrt(252)

        # Detect regime change: when rolling mean crosses certain thresholds
        df['roll_return'] = roll_mean_60d
        df['roll_volatility'] = roll_vol_60d

        # Regime classification
        regimes = []
        for i in range(252, len(df)):
            ret = roll_mean_60d.iloc[i]
            vol = roll_vol_60d.iloc[i]

            if ret > 0.15 and vol < 0.25:
                regime = "Bull - Low Vol"
            elif ret > 0.15 and vol >= 0.25:
                regime = "Bull - High Vol"
            elif ret < -0.05 and vol >= 0.30:
                regime = "Bear - High Vol"
            elif ret < -0.05:
                regime = "Bear - Low Vol"
            else:
                regime = "Sideways"

            # Check if regime changed from previous day
            if i > 252 and regimes and regime != regimes[-1]['regime']:
                regimes.append({
                    'date': df['date'].iloc[i],
                    'regime': regime,
                    'annual_return_est': round(ret * 100, 1),
                    'annual_volatility': round(vol * 100, 1),
                    'symbol': symbol or 'Unknown',
                    'is_change': True,
                })

        return regimes

    # ================================================================
    # Comprehensive Anomaly Report
    # ================================================================

    def full_anomaly_scan(self, df, symbol=None):
        """
        Run all anomaly detection methods and produce a comprehensive report.
        """
        if df is None or len(df) < 100:
            return {'symbol': symbol, 'status': 'Insufficient data'}

        print(f"\n  Scanning {symbol or 'stock'} for anomalies...")

        # Run all detectors
        price_anomalies = self.detect_price_anomalies(df, symbol)
        volume_anomalies = self.detect_volume_anomalies(df, symbol)
        vol_spikes = self.detect_volatility_spikes(df, symbol)
        regime_changes = self.detect_regime_changes(df, symbol)

        # Build report
        report = {
            'symbol': symbol or 'Unknown',
            'total_data_points': len(df),
            'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
            'price_anomalies': {
                'count': len(price_anomalies),
                'pct_of_days': round(len(price_anomalies) / len(df) * 100, 2),
                'extreme_up_days': len(price_anomalies[price_anomalies.get('anomaly_type') == 'extreme_up']) if len(price_anomalies) > 0 else 0,
                'extreme_down_days': len(price_anomalies[price_anomalies.get('anomaly_type') == 'extreme_down']) if len(price_anomalies) > 0 else 0,
            },
            'volume_anomalies': {
                'count': len(volume_anomalies),
                'pct_of_days': round(len(volume_anomalies) / len(df) * 100, 2),
            },
            'volatility_spikes': {
                'count': len(vol_spikes),
                'pct_of_days': round(len(vol_spikes) / len(df) * 100, 2),
            },
            'regime_changes': {
                'count': len(regime_changes),
                'recent_regimes': [r for r in regime_changes[-5:]] if regime_changes else [],
            },
        }

        # Overall anomaly severity
        total_anomaly_pct = (
            report['price_anomalies']['pct_of_days'] +
            report['volume_anomalies']['pct_of_days'] +
            report['volatility_spikes']['pct_of_days']
        ) / 3

        if total_anomaly_pct > 15:
            report['anomaly_severity'] = 'High'
        elif total_anomaly_pct > 5:
            report['anomaly_severity'] = 'Moderate'
        else:
            report['anomaly_severity'] = 'Low'

        return report

    # ================================================================
    # Market-Wide Anomaly Detection
    # ================================================================

    def detect_market_wide_events(self, stock_data_dict, date_range=None):
        """
        Find dates where multiple stocks simultaneously show anomalies.
        These likely correspond to market-wide events (e.g., 2008 crisis,
        COVID crash, election results, budget days).
        """
        if not stock_data_dict:
            return pd.DataFrame()

        # Collect anomaly dates for all stocks
        all_anomaly_dates = []
        for sym, df in stock_data_dict.items():
            anomalies = self.detect_price_anomalies(df, sym)
            if len(anomalies) > 0:
                all_anomaly_dates.append(anomalies[['date', 'symbol']])

        if not all_anomaly_dates:
            return pd.DataFrame()

        combined = pd.concat(all_anomaly_dates)

        # Count how many stocks had anomalies on each date
        date_counts = combined.groupby('date').size().reset_index(name='stock_count')
        date_counts = date_counts.sort_values('stock_count', ascending=False)

        # Calculate percentage of stocks affected
        total_stocks = len(stock_data_dict)
        date_counts['pct_stocks_affected'] = date_counts['stock_count'] / total_stocks * 100

        # Get the affected stocks for each date
        affected_stocks = combined.groupby('date')['symbol'].apply(list).reset_index()
        date_counts = date_counts.merge(affected_stocks, on='date')

        return date_counts
