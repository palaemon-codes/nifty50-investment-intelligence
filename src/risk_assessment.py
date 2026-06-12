"""
Risk Assessment Module
Evaluates risk metrics for individual stocks and portfolios.
Provides comprehensive risk analysis using multiple measures.

Metrics implemented:
- Volatility (annualized)
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Beta (market sensitivity)
- Calmar Ratio
- Omega Ratio
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import RISK_FREE_RATE


class RiskAssessor:
    """
    Comprehensive risk assessment for stocks and portfolios.
    Uses historical data to compute various risk metrics.
    """

    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.results_cache = {}

    # ================================================================
    # Individual Stock Risk Assessment
    # ================================================================

    def assess_stock(self, df, symbol=None):
        """
        Perform full risk assessment on a single stock.
        
        Parameters:
        -----------
        df : DataFrame with 'close' price column
        symbol : str, stock symbol for reporting
        
        Returns:
        --------
        dict with all risk metrics
        """
        if df is None or len(df) < 60:
            return self._empty_risk_report(symbol)

        close_prices = df['close'].values
        returns = pd.Series(close_prices).pct_change().dropna().values

        if len(returns) < 30:
            return self._empty_risk_report(symbol)

        # Compute all risk metrics
        volatility = self._annualized_volatility(returns)
        sharpe = self._sharpe_ratio(returns)
        sortino = self._sortino_ratio(returns)
        max_dd, max_dd_duration = self._max_drawdown(close_prices)
        var_95 = self._value_at_risk(returns, 0.95)
        cvar_95 = self._conditional_var(returns, 0.95)
        calmar = self._calmar_ratio(returns, max_dd)
        omega = self._omega_ratio(returns)

        # Additional statistics
        total_return = (close_prices[-1] / close_prices[0] - 1)
        positive_days = np.sum(returns > 0) / len(returns) * 100
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Risk category classification
        risk_category = self._classify_risk(volatility, max_dd)

        report = {
            'symbol': symbol or 'Unknown',
            'data_points': len(returns),
            'risk_category': risk_category,
            'volatility_annualized': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'max_drawdown_days': max_dd_duration,
            'var_95_daily': round(var_95 * 100, 3),
            'cvar_95_daily': round(cvar_95 * 100, 3),
            'calmar_ratio': round(calmar, 3),
            'omega_ratio': round(omega, 3),
            'total_return_pct': round(total_return * 100, 2),
            'positive_days_pct': round(positive_days, 1),
            'return_skewness': round(skewness, 3),
            'return_kurtosis': round(kurtosis, 3),
            'risk_free_rate_used': self.risk_free_rate,
        }

        self.results_cache[symbol or 'Unknown'] = report
        return report

    def assess_portfolio(self, returns_matrix, weights):
        """
        Assess risk for a portfolio given returns and weights.
        
        Parameters:
        -----------
        returns_matrix : DataFrame of stock returns (dates x symbols)
        weights : array of portfolio weights
        
        Returns:
        --------
        dict with portfolio risk metrics
        """
        if returns_matrix is None or len(returns_matrix) == 0:
            return {}

        # Calculate portfolio returns
        portfolio_returns = returns_matrix.dot(weights).dropna().values

        if len(portfolio_returns) < 30:
            return {}

        # Portfolio-level metrics
        volatility = self._annualized_volatility(portfolio_returns)
        sharpe = self._sharpe_ratio(portfolio_returns)
        sortino = self._sortino_ratio(portfolio_returns)

        # Portfolio cumulative value for drawdown
        cumulative = (1 + pd.Series(portfolio_returns)).cumprod().values
        max_dd, max_dd_duration = self._max_drawdown(cumulative)

        var_95 = self._value_at_risk(portfolio_returns, 0.95)
        cvar_95 = self._conditional_var(portfolio_returns, 0.95)
        calmar = self._calmar_ratio(portfolio_returns, max_dd)
        omega = self._omega_ratio(portfolio_returns)

        total_return = cumulative[-1] - 1

        return {
            'volatility_annualized': round(volatility * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'max_drawdown_days': max_dd_duration,
            'var_95_daily': round(var_95 * 100, 3),
            'cvar_95_daily': round(cvar_95 * 100, 3),
            'calmar_ratio': round(calmar, 3),
            'omega_ratio': round(omega, 3),
            'total_return_pct': round(total_return * 100, 2),
        }

    # ================================================================
    # Risk Metric Calculations
    # ================================================================

    def _annualized_volatility(self, returns, trading_days=252):
        """Annualized volatility from daily returns."""
        return np.std(returns) * np.sqrt(trading_days)

    def _sharpe_ratio(self, returns, trading_days=252):
        """
        Sharpe Ratio = (Mean Return - Risk Free Rate) / Std(Returns)
        Measures risk-adjusted return.
        """
        excess_returns = returns - self.risk_free_rate / trading_days
        if np.std(returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(trading_days)

    def _sortino_ratio(self, returns, trading_days=252):
        """
        Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Std
        Only penalizes downside volatility. Higher is better.
        """
        excess_returns = returns - self.risk_free_rate / trading_days
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        downside_std = np.std(downside_returns) * np.sqrt(trading_days)
        return np.mean(excess_returns) * trading_days / downside_std

    def _max_drawdown(self, prices):
        """
        Maximum Drawdown: Largest peak-to-trough decline.
        Returns (max_drawdown, max_drawdown_duration_in_days).
        """
        prices = np.array(prices)
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices - running_max) / running_max

        max_dd = abs(np.min(drawdowns))

        # Calculate drawdown duration
        max_dd_end = np.argmin(drawdowns)
        max_dd_start = np.argmax(prices[:max_dd_end + 1])
        max_dd_duration = max_dd_end - max_dd_start

        return max_dd, max_dd_duration

    def _value_at_risk(self, returns, confidence=0.95):
        """
        Value at Risk (Historical method).
        VaR at 95% confidence: The maximum loss expected 95% of the time.
        """
        return abs(np.percentile(returns, (1 - confidence) * 100))

    def _conditional_var(self, returns, confidence=0.95):
        """
        Conditional VaR (Expected Shortfall).
        Average loss beyond VaR threshold. More conservative than VaR.
        """
        var_threshold = self._value_at_risk(returns, confidence)
        beyond_var = returns[returns <= -var_threshold]
        if len(beyond_var) == 0:
            return var_threshold
        return abs(np.mean(beyond_var))

    def _calmar_ratio(self, returns, max_drawdown):
        """
        Calmar Ratio = Annualized Return / Maximum Drawdown.
        Higher values indicate better risk-adjusted performance.
        """
        annual_return = np.mean(returns) * 252
        if max_drawdown == 0:
            return 0
        return annual_return / max_drawdown

    def _omega_ratio(self, returns, threshold=0):
        """
        Omega Ratio = Probability of gains / Probability of losses.
        Ratio of gains above threshold to losses below threshold.
        Values > 1 indicate more upside than downside.
        """
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]

        if len(losses) == 0:
            return float('inf')
        if len(gains) == 0:
            return 0

        return np.sum(gains) / abs(np.sum(losses))

    def _classify_risk(self, volatility, max_drawdown):
        """
        Classify stock into risk category based on volatility and drawdown.
        """
        if volatility < 0.18 and max_drawdown < 0.30:
            return "Low Risk"
        elif volatility < 0.30 and max_drawdown < 0.50:
            return "Moderate Risk"
        elif volatility < 0.45:
            return "High Risk"
        else:
            return "Very High Risk"

    # ================================================================
    # Multi-Stock Comparison
    # ================================================================

    def compare_stocks(self, stock_data_dict, top_n=20):
        """
        Compare risk metrics across multiple stocks.
        Returns a DataFrame with risk metrics for each stock.
        """
        results = []
        for sym, df in stock_data_dict.items():
            report = self.assess_stock(df, sym)
            if report['sharpe_ratio'] != 0:
                results.append(report)

        if not results:
            return pd.DataFrame()

        df_results = pd.DataFrame(results)
        return df_results.sort_values('sharpe_ratio', ascending=False).head(top_n)

    def get_risk_summary_text(self, report):
        """
        Generate a human-readable risk summary from a risk report.
        """
        if not report or report.get('volatility_annualized') == 0:
            return "Insufficient data for risk analysis."

        category = report['risk_category']
        vol = report['volatility_annualized']
        sharpe = report['sharpe_ratio']
        max_dd = report['max_drawdown_pct']

        summary = f"Risk Category: **{category}**\n\n"

        # Volatility interpretation
        if vol < 15:
            summary += f"- Annual volatility of {vol}% is relatively low, suggesting stable price movements.\n"
        elif vol < 25:
            summary += f"- Annual volatility of {vol}% is moderate, typical for large-cap stocks.\n"
        elif vol < 35:
            summary += f"- Annual volatility of {vol}% is elevated, indicating significant price swings.\n"
        else:
            summary += f"- Annual volatility of {vol}% is high, suggesting substantial price fluctuations.\n"

        # Sharpe ratio interpretation
        if sharpe > 1.0:
            summary += f"- Sharpe Ratio of {sharpe} indicates good risk-adjusted returns.\n"
        elif sharpe > 0.5:
            summary += f"- Sharpe Ratio of {sharpe} indicates adequate risk-adjusted returns.\n"
        elif sharpe > 0:
            summary += f"- Sharpe Ratio of {sharpe} suggests returns barely compensate for the risk taken.\n"
        else:
            summary += f"- Negative Sharpe Ratio of {sharpe} indicates returns below the risk-free rate.\n"

        # Drawdown interpretation
        if max_dd < 20:
            summary += f"- Maximum drawdown of {max_dd}% is manageable for most investors.\n"
        elif max_dd < 40:
            summary += f"- Maximum drawdown of {max_dd}% requires moderate risk tolerance.\n"
        else:
            summary += f"- Maximum drawdown of {max_dd}% suggests significant downside risk.\n"

        return summary

    # ================================================================
    # Rolling Risk Metrics (for trend analysis)
    # ================================================================

    def rolling_volatility(self, df, window=60):
        """Calculate rolling annualized volatility."""
        returns = df['close'].pct_change().dropna()
        return returns.rolling(window=window).std() * np.sqrt(252)

    def rolling_sharpe(self, df, window=252):
        """Calculate rolling Sharpe ratio."""
        returns = df['close'].pct_change().dropna()
        daily_rf = self.risk_free_rate / 252
        excess = returns - daily_rf
        rolling_sharpe = (
            excess.rolling(window=window).mean() /
            returns.rolling(window=window).std()
        ) * np.sqrt(252)
        return rolling_sharpe

    def rolling_var(self, df, window=252, confidence=0.95):
        """Calculate rolling Value at Risk."""
        returns = df['close'].pct_change().dropna()
        return returns.rolling(window=window).quantile(1 - confidence).abs()

    # ================================================================
    # Helper
    # ================================================================

    def _empty_risk_report(self, symbol=None):
        """Return empty report when data is insufficient."""
        return {
            'symbol': symbol or 'Unknown',
            'data_points': 0,
            'risk_category': 'Unknown',
            'volatility_annualized': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_days': 0,
            'var_95_daily': 0,
            'cvar_95_daily': 0,
            'calmar_ratio': 0,
            'omega_ratio': 0,
            'total_return_pct': 0,
            'positive_days_pct': 0,
            'return_skewness': 0,
            'return_kurtosis': 0,
            'risk_free_rate_used': self.risk_free_rate,
        }
