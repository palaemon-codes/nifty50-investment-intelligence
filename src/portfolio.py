"""
Portfolio Construction Module
Builds optimized investment portfolios for different investor profiles:
- Conservative (low risk, stable returns)
- Balanced (moderate risk and returns)
- Aggressive (higher risk, growth-oriented)

Uses Modern Portfolio Theory, Efficient Frontier, and risk-based allocation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from config import (
    RISK_FREE_RATE, PORTFOLIO_TOP_N,
    CONSERVATIVE_MAX_VOLATILITY, BALANCED_MAX_VOLATILITY, AGGRESSIVE_MAX_VOLATILITY,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


class PortfolioConstructor:
    """
    Builds investment portfolios using Modern Portfolio Theory.
    Supports multiple investor profiles with different risk tolerances.
    """

    def __init__(self, stock_data_dict, sector_mapping=None):
        """
        Initialize with stock data.
        
        Parameters:
        -----------
        stock_data_dict : dict, symbol -> DataFrame with at least 'close' and 'date' columns
        sector_mapping : dict, symbol -> sector name
        """
        self.stock_data = stock_data_dict
        self.sector_mapping = sector_mapping or {}
        self.returns_matrix = None
        self.mean_returns = None
        self.cov_matrix = None
        self.symbols = []
        self._prepare_returns_data()

    def _prepare_returns_data(self):
        """Build returns matrix from all stocks."""
        if not self.stock_data:
            return

        returns_dict = {}
        valid_symbols = []

        for sym, df in self.stock_data.items():
            if df is None or len(df) < 252:  # At least 1 year of data
                continue
            # Calculate daily returns
            returns = df.set_index('date')['close'].pct_change().dropna()
            if len(returns) >= 252:
                returns_dict[sym] = returns
                valid_symbols.append(sym)

        if not returns_dict:
            print("  No stocks with sufficient data for portfolio construction")
            return

        # Align all returns to common date index
        self.returns_matrix = pd.DataFrame(returns_dict).dropna()

        if len(self.returns_matrix) < 60:
            print("  Not enough overlapping data points")
            return

        self.mean_returns = self.returns_matrix.mean() * 252  # Annualized
        self.cov_matrix = self.returns_matrix.cov() * 252      # Annualized
        self.symbols = list(self.returns_matrix.columns)

        print(f"  Portfolio universe: {len(self.symbols)} stocks with {len(self.returns_matrix)} trading days")

    # ================================================================
    # Portfolio Metrics
    # ================================================================

    def portfolio_return(self, weights):
        """Calculate expected annual return for given weights."""
        return np.sum(self.mean_returns * weights)

    def portfolio_volatility(self, weights):
        """Calculate annualized portfolio volatility."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def portfolio_sharpe_ratio(self, weights):
        """Calculate Sharpe Ratio for given weights."""
        p_return = self.portfolio_return(weights)
        p_vol = self.portfolio_volatility(weights)
        if p_vol == 0:
            return 0
        return (p_return - RISK_FREE_RATE) / p_vol

    def portfolio_sortino_ratio(self, weights):
        """Calculate Sortino Ratio (uses downside deviation only)."""
        portfolio_returns = self.returns_matrix.dot(weights)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

        p_return = self.portfolio_return(weights)
        if downside_deviation == 0:
            return 0
        return (p_return - RISK_FREE_RATE) / downside_deviation

    def max_drawdown(self, weights):
        """Calculate maximum drawdown for a portfolio."""
        portfolio_returns = self.returns_matrix.dot(weights)
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    # ================================================================
    # Optimization Functions
    # ================================================================

    def _neg_sharpe_ratio(self, weights):
        """Negative Sharpe Ratio (for minimization)."""
        return -self.portfolio_sharpe_ratio(weights)

    def _portfolio_variance(self, weights):
        """Portfolio variance (for minimization)."""
        return self.portfolio_volatility(weights) ** 2

    def _check_constraints(self, weights):
        """Check that weights sum to 1."""
        return np.sum(weights) - 1

    def optimize_max_sharpe(self):
        """
        Find the portfolio with maximum Sharpe Ratio (tangency portfolio).
        """
        n = len(self.symbols)
        initial_weights = np.ones(n) / n
        bounds = tuple((0, 0.35) for _ in range(n))  # Max 35% per stock

        constraints = {'type': 'eq', 'fun': self._check_constraints}

        result = minimize(
            self._neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-12},
        )

        if result.success:
            weights = result.x
            # Zero out very small weights
            weights[weights < 0.005] = 0
            weights = weights / weights.sum()
            return weights
        else:
            print(f"  Optimization warning: {result.message}")
            return initial_weights

    def optimize_min_volatility(self):
        """Find the minimum variance portfolio."""
        n = len(self.symbols)
        initial_weights = np.ones(n) / n
        bounds = tuple((0, 0.30) for _ in range(n))
        constraints = {'type': 'eq', 'fun': self._check_constraints}

        result = minimize(
            self._portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-12},
        )

        if result.success:
            weights = result.x
            weights[weights < 0.005] = 0
            weights = weights / weights.sum()
            return weights
        return initial_weights

    def optimize_target_volatility(self, target_vol):
        """
        Find portfolio closest to target volatility while maximizing return.
        """
        n = len(self.symbols)
        initial_weights = np.ones(n) / n
        bounds = tuple((0, 0.40) for _ in range(n))

        def objective(weights):
            # Maximize return, penalize deviation from target vol
            ret = -self.portfolio_return(weights)
            vol_diff = abs(self.portfolio_volatility(weights) - target_vol)
            return ret + 10 * vol_diff

        constraints = {'type': 'eq', 'fun': self._check_constraints}

        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-12},
        )

        if result.success:
            weights = result.x
            weights[weights < 0.005] = 0
            weights = weights / weights.sum()
            return weights
        return initial_weights

    # ================================================================
    # Investor Profile Portfolios
    # ================================================================

    def build_conservative_portfolio(self):
        """
        Build a conservative portfolio:
        - Focus on low volatility and capital preservation
        - Prefer large-cap, stable companies
        - Target: 8-12% annual return, <15% volatility
        """
        if self.returns_matrix is None or len(self.symbols) < 3:
            return self._empty_portfolio("Conservative")

        print("\n  Building Conservative Portfolio...")
        weights = self.optimize_min_volatility()

        # Create allocation details
        allocation = self._format_allocation(weights)

        # Calculate metrics
        p_return = self.portfolio_return(weights)
        p_vol = self.portfolio_volatility(weights)
        sharpe = self.portfolio_sharpe_ratio(weights)
        sortino = self.portfolio_sortino_ratio(weights)
        max_dd = self.max_drawdown(weights)

        # Get sector diversification
        sectors = self._get_sector_allocation(weights)

        portfolio = {
            'profile': 'Conservative',
            'description': (
                "This portfolio is designed for investors who prioritize capital preservation "
                "and stable returns over aggressive growth. It focuses on low-volatility stocks "
                "with strong fundamentals and consistent performance history."
            ),
            'allocation': allocation,
            'metrics': {
                'Expected Annual Return': f"{p_return * 100:.2f}%",
                'Expected Volatility': f"{p_vol * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Sortino Ratio': f"{sortino:.3f}",
                'Maximum Drawdown': f"{max_dd * 100:.2f}%",
                'Risk-Free Rate Used': f"{RISK_FREE_RATE * 100:.1f}%",
            },
            'sector_allocation': sectors,
            'investment_horizon': '3-5+ years',
            'rebalancing_frequency': 'Quarterly',
            'risk_level': 'Low',
        }

        return portfolio

    def build_balanced_portfolio(self):
        """
        Build a balanced portfolio:
        - Mix of growth and value stocks
        - Moderate risk with diversification across sectors
        - Target: 12-18% annual return, 15-25% volatility
        """
        if self.returns_matrix is None or len(self.symbols) < 3:
            return self._empty_portfolio("Balanced")

        print("\n  Building Balanced Portfolio...")
        weights = self.optimize_target_volatility(0.20)

        allocation = self._format_allocation(weights)

        p_return = self.portfolio_return(weights)
        p_vol = self.portfolio_volatility(weights)
        sharpe = self.portfolio_sharpe_ratio(weights)
        sortino = self.portfolio_sortino_ratio(weights)
        max_dd = self.max_drawdown(weights)

        sectors = self._get_sector_allocation(weights)

        portfolio = {
            'profile': 'Balanced',
            'description': (
                "This portfolio strikes a balance between growth and stability. "
                "It diversifies across sectors and includes a mix of established companies "
                "and growth-oriented stocks. Suitable for investors with a moderate risk appetite "
                "who want reasonable returns without excessive volatility."
            ),
            'allocation': allocation,
            'metrics': {
                'Expected Annual Return': f"{p_return * 100:.2f}%",
                'Expected Volatility': f"{p_vol * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Sortino Ratio': f"{sortino:.3f}",
                'Maximum Drawdown': f"{max_dd * 100:.2f}%",
                'Risk-Free Rate Used': f"{RISK_FREE_RATE * 100:.1f}%",
            },
            'sector_allocation': sectors,
            'investment_horizon': '2-4 years',
            'rebalancing_frequency': 'Quarterly',
            'risk_level': 'Moderate',
        }

        return portfolio

    def build_aggressive_portfolio(self):
        """
        Build an aggressive portfolio:
        - Focus on high growth potential
        - Higher risk tolerance
        - Target: 18%+ annual return, 25-40% volatility
        """
        if self.returns_matrix is None or len(self.symbols) < 3:
            return self._empty_portfolio("Aggressive")

        print("\n  Building Aggressive Portfolio...")
        weights = self.optimize_max_sharpe()

        allocation = self._format_allocation(weights)

        p_return = self.portfolio_return(weights)
        p_vol = self.portfolio_volatility(weights)
        sharpe = self.portfolio_sharpe_ratio(weights)
        sortino = self.portfolio_sortino_ratio(weights)
        max_dd = self.max_drawdown(weights)

        sectors = self._get_sector_allocation(weights)

        portfolio = {
            'profile': 'Aggressive',
            'description': (
                "This portfolio is built for investors seeking maximum growth potential "
                "and willing to accept higher volatility and larger drawdowns. It focuses on "
                "high-momentum stocks and sectors with strong growth trajectories. "
                "Not suitable for risk-averse investors or short-term goals."
            ),
            'allocation': allocation,
            'metrics': {
                'Expected Annual Return': f"{p_return * 100:.2f}%",
                'Expected Volatility': f"{p_vol * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe:.3f}",
                'Sortino Ratio': f"{sortino:.3f}",
                'Maximum Drawdown': f"{max_dd * 100:.2f}%",
                'Risk-Free Rate Used': f"{RISK_FREE_RATE * 100:.1f}%",
            },
            'sector_allocation': sectors,
            'investment_horizon': '1-3 years',
            'rebalancing_frequency': 'Monthly',
            'risk_level': 'High',
        }

        return portfolio

    def build_all_portfolios(self):
        """Build all three investor profile portfolios."""
        return {
            'Conservative': self.build_conservative_portfolio(),
            'Balanced': self.build_balanced_portfolio(),
            'Aggressive': self.build_aggressive_portfolio(),
        }

    # ================================================================
    # Efficient Frontier
    # ================================================================

    def compute_efficient_frontier(self, n_points=100):
        """
        Compute the Efficient Frontier - set of optimal portfolios
        offering the highest expected return for a given level of risk.
        """
        if self.returns_matrix is None or len(self.symbols) < 2:
            return [], [], []

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []

        # Find min and max volatility portfolios
        min_vol_weights = self.optimize_min_volatility()
        max_sharpe_weights = self.optimize_max_sharpe()

        min_vol = self.portfolio_volatility(min_vol_weights)
        max_vol = self.portfolio_volatility(max_sharpe_weights) * 1.5

        target_vols = np.linspace(min_vol, max_vol, n_points)

        for target in target_vols:
            try:
                weights = self.optimize_target_volatility(target)
                ret = self.portfolio_return(weights)
                vol = self.portfolio_volatility(weights)

                frontier_returns.append(ret)
                frontier_volatilities.append(vol)
                frontier_weights.append(weights)
            except Exception:
                continue

        return frontier_returns, frontier_volatilities, frontier_weights

    # ================================================================
    # Helper Methods
    # ================================================================

    def _format_allocation(self, weights, min_weight=0.005):
        """
        Format portfolio weights into a sorted list of allocations.
        Filters out very small positions.
        """
        allocation = []
        for sym, weight in zip(self.symbols, weights):
            if weight >= min_weight:
                sector = self.sector_mapping.get(sym, "Unknown")
                allocation.append({
                    'symbol': sym,
                    'weight': round(weight * 100, 1),
                    'sector': sector,
                })

        # Sort by weight descending
        allocation.sort(key=lambda x: x['weight'], reverse=True)

        # Limit to top N holdings
        allocation = allocation[:PORTFOLIO_TOP_N]

        # Re-normalize to 100%
        total = sum(a['weight'] for a in allocation)
        if total > 0:
            for a in allocation:
                a['weight'] = round(a['weight'] / total * 100, 1)

        return allocation

    def _get_sector_allocation(self, weights):
        """Calculate allocation percentage per sector."""
        sector_weights = {}
        for sym, weight in zip(self.symbols, weights):
            if weight > 0.001:
                sector = self.sector_mapping.get(sym, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

        # Convert to percentages
        total = sum(sector_weights.values())
        return {k: round(v / total * 100, 1) for k, v in sector_weights.items()}

    def _empty_portfolio(self, profile_name):
        """Return empty portfolio structure when data is insufficient."""
        return {
            'profile': profile_name,
            'description': 'Insufficient data to build this portfolio. Please load more stock data.',
            'allocation': [],
            'metrics': {
                'Expected Annual Return': 'N/A',
                'Expected Volatility': 'N/A',
                'Sharpe Ratio': 'N/A',
                'Sortino Ratio': 'N/A',
                'Maximum Drawdown': 'N/A',
                'Risk-Free Rate Used': f"{RISK_FREE_RATE * 100:.1f}%",
            },
            'sector_allocation': {},
            'investment_horizon': 'N/A',
            'rebalancing_frequency': 'N/A',
            'risk_level': 'N/A',
        }

    def get_stock_recommendation_reasoning(self, symbol, weight):
        """
        Generate a human-readable justification for including a stock in the portfolio.
        """
        df = self.stock_data.get(symbol)
        if df is None or len(df) < 252:
            return f"{symbol}: Included for diversification purposes."

        reasons = []

        # Check returns trend
        returns = df['close'].pct_change().dropna()
        recent_return = (df['close'].iloc[-1] / df['close'].iloc[-252] - 1) if len(df) >= 252 else 0

        if recent_return > 0.20:
            reasons.append("strong recent performance")
        elif recent_return > 0:
            reasons.append("positive momentum")

        # Check volatility
        annual_vol = returns.std() * np.sqrt(252)
        if annual_vol < 0.15:
            reasons.append("low volatility")
        elif annual_vol < 0.25:
            reasons.append("moderate volatility")

        # Check for consistent returns
        positive_days = (returns > 0).sum() / len(returns)
        if positive_days > 0.52:
            reasons.append("consistent positive daily returns")

        sector = self.sector_mapping.get(symbol, "its sector")
        reasons.append(f"exposure to {sector} sector")

        if not reasons:
            return f"{symbol}: Included based on portfolio optimization results."

        return f"{symbol}: Selected for {' and '.join(reasons[:3])}."
