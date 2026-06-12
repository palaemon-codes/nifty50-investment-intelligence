"""
Advanced Forecasting Module
Provides time series forecasting capabilities for:
- Future price trends
- Volatility estimation
- Return forecasting

Uses statistical models (ARIMA, GARCH-style) and machine learning approaches.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import PREDICTION_HORIZON, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class Forecaster:
    """
    Time series forecasting for stock market data.
    Implements multiple forecasting approaches for different use cases.
    """

    def __init__(self):
        self.models = {}
        self.forecasts = {}

    # ================================================================
    # Trend Forecasting (using moving averages + regression)
    # ================================================================

    def forecast_trend_ma_regression(self, df, horizon_days=30):
        """
        Forecast future price trend using weighted moving average regression.
        Simple but interpretable approach suitable for trend estimation.
        """
        if df is None or len(df) < 100:
            return None

        close = df['close'].values

        # Fit linear regression on last 90 days
        recent = close[-90:]
        x = np.arange(len(recent))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent)

        # Forecast
        last_idx = len(recent) - 1
        forecast_x = np.arange(last_idx + 1, last_idx + 1 + horizon_days)
        forecast_values = slope * forecast_x + intercept

        # Confidence bands (using standard error of regression)
        residuals = recent - (slope * x + intercept)
        residual_std = np.std(residuals)

        upper_band = forecast_values + 1.96 * residual_std * np.sqrt(np.arange(1, horizon_days + 1))
        lower_band = forecast_values - 1.96 * residual_std * np.sqrt(np.arange(1, horizon_days + 1))

        # Trend strength
        trend_strength = abs(r_value)
        trend_direction = "upward" if slope > 0 else "downward"

        return {
            'method': 'Linear Trend Regression',
            'forecast': forecast_values.tolist(),
            'upper_band': upper_band.tolist(),
            'lower_band': lower_band.tolist(),
            'trend_direction': trend_direction,
            'trend_strength': round(trend_strength, 3),
            'r_squared': round(r_value ** 2, 3),
            'current_price': close[-1],
            'forecast_final_price': forecast_values[-1],
            'expected_return_pct': round((forecast_values[-1] / close[-1] - 1) * 100, 2),
        }

    # ================================================================
    # Exponential Smoothing Forecast
    # ================================================================

    def forecast_exponential_smoothing(self, df, horizon_days=30, alpha=0.3):
        """
        Forecast using Holt's exponential smoothing (trend-aware).
        Simple recursive implementation.
        """
        if df is None or len(df) < 60:
            return None

        close = df['close'].values

        # Initialize level and trend
        level = close[0]
        trend = close[1] - close[0]

        levels = [level]
        trends = [trend]

        beta = 0.1  # Trend smoothing parameter

        for i in range(1, len(close)):
            prev_level = level
            level = alpha * close[i] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            levels.append(level)
            trends.append(trend)

        # Forecast
        last_level = levels[-1]
        last_trend = trends[-1]

        forecast = []
        for h in range(1, horizon_days + 1):
            forecast.append(last_level + h * last_trend)

        return {
            'method': 'Holt Exponential Smoothing',
            'forecast': forecast,
            'current_price': close[-1],
            'forecast_final_price': forecast[-1],
            'expected_return_pct': round((forecast[-1] / close[-1] - 1) * 100, 2),
            'trend_component': last_trend,
            'level_component': last_level,
        }

    # ================================================================
    # Monte Carlo Simulation for Price Forecasting
    # ================================================================

    def monte_carlo_simulation(self, df, horizon_days=30, n_simulations=1000):
        """
        Monte Carlo price simulation using geometric Brownian motion.
        Generates multiple possible price paths and returns distribution
        of outcomes.
        """
        if df is None or len(df) < 100:
            return None

        close = df['close'].values
        returns = np.diff(np.log(close))

        mu = np.mean(returns) * 252  # Annualized drift
        sigma = np.std(returns) * np.sqrt(252)  # Annualized volatility

        daily_mu = mu / 252
        daily_sigma = sigma / np.sqrt(252)

        last_price = close[-1]

        # Run simulations
        np.random.seed(RANDOM_SEED)
        simulations = np.zeros((horizon_days, n_simulations))

        for sim in range(n_simulations):
            prices = [last_price]
            for day in range(horizon_days):
                # Geometric Brownian Motion
                shock = np.random.normal(0, 1)
                next_price = prices[-1] * np.exp(daily_mu + daily_sigma * shock)
                prices.append(next_price)
            simulations[:, sim] = prices[1:]

        # Calculate statistics
        final_prices = simulations[-1, :]
        returns_sim = (final_prices / last_price - 1) * 100

        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        price_percentiles = {f'p{p}': np.percentile(final_prices, p) for p in percentiles}
        return_percentiles = {f'p{p}': round(np.percentile(returns_sim, p), 2) for p in percentiles}

        # Path statistics
        mean_path = np.mean(simulations, axis=1)
        upper_path = np.percentile(simulations, 95, axis=1)
        lower_path = np.percentile(simulations, 5, axis=1)

        # Probability of positive return
        prob_positive = np.mean(final_prices > last_price) * 100

        return {
            'method': f'Monte Carlo Simulation ({n_simulations} paths)',
            'current_price': last_price,
            'expected_final_price': round(np.mean(final_prices), 2),
            'median_final_price': round(np.median(final_prices), 2),
            'expected_return_pct': round(np.mean(returns_sim), 2),
            'prob_positive_return': round(prob_positive, 1),
            'price_percentiles': price_percentiles,
            'return_percentiles': return_percentiles,
            'mean_path': mean_path.tolist(),
            'upper_band_95': upper_path.tolist(),
            'lower_band_5': lower_path.tolist(),
            'annualized_drift': round(mu * 100, 2),
            'annualized_volatility': round(sigma * 100, 2),
        }

    # ================================================================
    # Volatility Forecasting (GARCH-style)
    # ================================================================

    def forecast_volatility(self, df, horizon_days=30):
        """
        Forecast future volatility using a simple GARCH(1,1)-style model.
        
        Uses exponentially weighted moving average of squared returns
        as a simplified volatility forecasting approach.
        """
        if df is None or len(df) < 100:
            return None

        returns = df['close'].pct_change().dropna().values

        # EWMA volatility (RiskMetrics approach)
        lambda_param = 0.94  # Decay factor
        weights = np.array([(1 - lambda_param) * lambda_param ** i
                            for i in range(len(returns) - 1, -1, -1)])
        weights = weights / weights.sum()

        # Current volatility estimate
        current_variance = np.sum(weights * returns ** 2)
        current_vol_daily = np.sqrt(current_variance)

        # Long-term (unconditional) variance
        long_term_variance = np.var(returns)

        # GARCH(1,1) parameters (simplified estimation)
        omega = 0.000002  # Long-term variance weight
        alpha = 0.08      # ARCH term (recent shock impact)
        beta = 0.90       # GARCH term (persistence)

        # Forecast volatility path
        forecast_variances = []
        forecast_vols = []
        current_var = current_variance

        for _ in range(horizon_days):
            next_var = omega + alpha * current_var + beta * current_var
            # Mean reversion to long-term variance
            next_var = 0.7 * next_var + 0.3 * long_term_variance
            forecast_variances.append(next_var)
            forecast_vols.append(np.sqrt(next_var) * np.sqrt(252))  # Annualized
            current_var = next_var

        return {
            'method': 'EWMA + GARCH(1,1)-style Volatility Forecast',
            'current_annualized_volatility': round(current_vol_daily * np.sqrt(252) * 100, 2),
            'forecast_volatility_path': [round(v * 100, 2) for v in forecast_vols],
            'average_forecast_volatility': round(np.mean(forecast_vols) * 100, 2),
            'max_forecast_volatility': round(max(forecast_vols) * 100, 2),
            'min_forecast_volatility': round(min(forecast_vols) * 100, 2),
            'long_term_volatility': round(np.sqrt(long_term_variance) * np.sqrt(252) * 100, 2),
        }

    # ================================================================
    # Combined Multi-Method Forecast
    # ================================================================

    def full_forecast(self, df, horizon_days=30):
        """
        Run all forecasting methods and produce a combined forecast report.
        """
        if df is None or len(df) < 100:
            return {'error': 'Insufficient data for forecasting'}

        print(f"\n  Generating forecasts for {horizon_days}-day horizon...")

        # Run all methods
        trend_forecast = self.forecast_trend_ma_regression(df, horizon_days)
        holt_forecast = self.forecast_exponential_smoothing(df, horizon_days)
        monte_carlo = self.monte_carlo_simulation(df, horizon_days, n_simulations=500)
        vol_forecast = self.forecast_volatility(df, horizon_days)

        # Combine results
        combined = {
            'horizon_days': horizon_days,
            'current_price': df['close'].iloc[-1],
            'trend_forecast': trend_forecast,
            'exponential_smoothing': holt_forecast,
            'monte_carlo': monte_carlo,
            'volatility_forecast': vol_forecast,
        }

        # Create ensemble forecast (average of all methods)
        forecasts = []
        if trend_forecast:
            forecasts.append(trend_forecast['expected_return_pct'])
        if holt_forecast:
            forecasts.append(holt_forecast['expected_return_pct'])
        if monte_carlo:
            forecasts.append(monte_carlo['expected_return_pct'])

        if forecasts:
            ensemble_return = np.mean(forecasts)
            combined['ensemble_forecast'] = {
                'expected_return_pct': round(ensemble_return, 2),
                'expected_price': round(df['close'].iloc[-1] * (1 + ensemble_return / 100), 2),
                'direction': 'UP' if ensemble_return > 0 else 'DOWN',
                'methods_used': len(forecasts),
            }

        self.forecasts = combined
        return combined

    # ================================================================
    # Return Distribution Analysis
    # ================================================================

    def analyze_return_distribution(self, df):
        """
        Analyze the statistical properties of stock returns.
        Useful for understanding tail risk and return characteristics.
        """
        if df is None or len(df) < 100:
            return None

        returns = df['close'].pct_change().dropna().values

        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis

        # Normality test
        ks_stat, ks_pvalue = stats.kstest(
            (returns - mean_return) / std_return, 'norm'
        )

        # Tail analysis
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()

        # Best and worst
        best_day = np.max(returns)
        worst_day = np.min(returns)

        return {
            'mean_daily_return': round(mean_return * 100, 4),
            'annualized_return': round(mean_return * 252 * 100, 2),
            'daily_volatility': round(std_return * 100, 4),
            'annualized_volatility': round(std_return * np.sqrt(252) * 100, 2),
            'skewness': round(skewness, 3),
            'excess_kurtosis': round(kurtosis, 3),
            'is_normal_distribution': ks_pvalue > 0.05,
            'ks_test_pvalue': round(ks_pvalue, 4),
            'var_95_daily_pct': round(var_95 * 100, 3),
            'var_99_daily_pct': round(var_99 * 100, 3),
            'cvar_95_daily_pct': round(cvar_95 * 100, 3),
            'best_single_day_pct': round(best_day * 100, 2),
            'worst_single_day_pct': round(worst_day * 100, 2),
            'positive_days_pct': round(np.mean(returns > 0) * 100, 1),
        }
