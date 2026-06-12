"""
Explainable AI (XAI) Framework
Provides human-understandable explanations for model predictions and recommendations.

Techniques used:
- SHAP (SHapley Additive exPlanations) for feature importance
- LIME-style local explanations
- Natural language explanation generation
- Decision path visualization data
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class ExplainableAI:
    """
    Generates explanations for stock predictions and investment recommendations.
    Makes AI decisions transparent and understandable to users.
    """

    def __init__(self):
        self.explanations = {}

    # ================================================================
    # Prediction Explanation
    # ================================================================

    def explain_prediction(self, stock_symbol, predicted_return, features_used,
                           feature_values, feature_importance=None):
        """
        Generate a natural language explanation for a stock prediction.
        
        Parameters:
        -----------
        stock_symbol : str
        predicted_return : float, predicted return percentage
        features_used : list of feature names
        feature_values : dict, current values of key features
        feature_importance : list of (feature, importance_score), optional
        """
        direction = "upward" if predicted_return > 0 else "downward"
        strength = abs(predicted_return)

        if strength < 0.01:
            confidence = "low confidence"
            movement_desc = "relatively flat movement"
        elif strength < 0.03:
            confidence = "moderate confidence"
            movement_desc = f"a {direction} movement"
        elif strength < 0.07:
            confidence = "high confidence"
            movement_desc = f"a significant {direction} trend"
        else:
            confidence = "very high confidence"
            movement_desc = f"a strong {direction} trend"

        # Build explanation
        explanation = {
            'summary': (
                f"Our model predicts {movement_desc} for **{stock_symbol}** with "
                f"**{confidence}**. The expected return over the forecast horizon "
                f"is approximately **{predicted_return * 100:.2f}%**."
            ),
            'direction': direction,
            'confidence': confidence,
            'predicted_return_pct': round(predicted_return * 100, 2),
        }

        # Add feature-level explanations if importance is available
        if feature_importance:
            top_features = sorted(feature_importance, key=lambda x: x[1], reverse=True)[:5]
            feature_explanations = []

            for feat_name, importance in top_features:
                feat_value = feature_values.get(feat_name, None)
                feat_desc = self._describe_feature(feat_name, feat_value, direction)
                feature_explanations.append({
                    'feature': feat_name,
                    'importance': round(importance * 100, 1),
                    'description': feat_desc,
                })

            explanation['top_factors'] = feature_explanations

        self.explanations[stock_symbol] = explanation
        return explanation

    def _describe_feature(self, feature_name, value, direction):
        """
        Convert a feature name and value into a human-readable description.
        """
        descriptions = {
            'rsi': lambda v, d: (
                f"RSI is at {v:.1f}, indicating "
                f"{'overbought conditions' if v > 70 else 'oversold conditions' if v < 30 else 'neutral momentum'}"
            ),
            'macd': lambda v, d: (
                f"MACD is {'positive (bullish signal)' if v > 0 else 'negative (bearish signal)'}"
            ),
            'macd_histogram': lambda v, d: (
                f"MACD histogram is {'rising (increasing momentum)' if v > 0 else 'falling (decreasing momentum)'}"
            ),
            'volume_ratio': lambda v, d: (
                f"Trading volume is {v:.1f}x the average, suggesting "
                f"{'unusually high activity' if v > 2 else 'normal trading activity' if v > 0.5 else 'low activity'}"
            ),
            'bb_position': lambda v, d: (
                f"Price is at the {v:.0%} level of Bollinger Bands "
                f"({'near upper band - potentially overbought' if v > 0.8 else 'near lower band - potentially oversold' if v < 0.2 else 'in the middle range'})"
            ),
            'volatility_20d': lambda v, d: (
                f"20-day volatility is {v*100:.1f}%, "
                f"{'elevated compared to normal levels' if v > 0.03 else 'within normal range'}"
            ),
            'sma_50': lambda v, d: (
                f"50-day moving average context"
            ),
            'price_to_sma_200': lambda v, d: (
                f"Price is {v*100:.1f}% {'above' if v > 0 else 'below'} the 200-day MA "
                f"({'long-term uptrend' if v > 0.05 else 'long-term downtrend' if v < -0.05 else 'near long-term average'})"
            ),
            'daily_return': lambda v, d: (
                f"Recent daily return is {v*100:.2f}%"
            ),
        }

        # Try exact match
        for key, desc_func in descriptions.items():
            if key in feature_name.lower():
                try:
                    return desc_func(value, direction)
                except Exception:
                    pass

        # Generic description based on feature name
        if 'return' in feature_name.lower():
            return f"{feature_name} shows {'positive' if value > 0 else 'negative'} momentum at {value*100:.2f}%"
        elif 'volume' in feature_name.lower():
            return f"{feature_name} indicates trading activity level"
        elif 'volatility' in feature_name.lower():
            return f"{feature_name} measures price fluctuation intensity"
        elif 'ma' in feature_name.lower() or 'sma' in feature_name.lower():
            return f"{feature_name} shows the trend direction context"

        return f"{feature_name} contributes to the overall prediction"

    # ================================================================
    # Portfolio Recommendation Explanation
    # ================================================================

    def explain_portfolio(self, portfolio, stock_data_dict, sector_mapping=None):
        """
        Generate comprehensive explanation for a portfolio recommendation.
        """
        profile = portfolio.get('profile', 'Custom')
        metrics = portfolio.get('metrics', {})
        allocation = portfolio.get('allocation', [])

        # Overall strategy explanation
        strategy_explanations = {
            'Conservative': (
                "This conservative portfolio prioritizes **capital preservation** over aggressive growth. "
                "We selected stocks with lower volatility, consistent earnings history, and strong balance sheets. "
                "The allocation is diversified across defensive sectors to reduce downside risk during market downturns. "
                "The optimization target was to minimize portfolio volatility while maintaining positive expected returns."
            ),
            'Balanced': (
                "This balanced portfolio aims for **steady growth with controlled risk**. "
                "The allocation combines stable large-cap stocks with selected growth opportunities. "
                "We optimized for a target volatility of ~20% while maximizing the Sharpe Ratio. "
                "Sector diversification helps protect against sector-specific downturns while capturing upside."
            ),
            'Aggressive': (
                "This aggressive portfolio is designed for **maximum growth potential**. "
                "It focuses on high-momentum stocks and growth sectors, accepting higher volatility "
                "in exchange for potentially higher returns. The optimization maximizes the Sharpe Ratio, "
                "seeking the best risk-adjusted returns. This approach is suitable for investors with "
                "a long time horizon who can tolerate significant short-term fluctuations."
            ),
        }

        strategy = strategy_explanations.get(
            profile,
            "This portfolio was constructed using Modern Portfolio Theory optimization."
        )

        # Stock-level reasoning
        stock_reasons = []
        for item in allocation[:8]:
            sym = item['symbol']
            weight = item['weight']
            sector = item.get('sector', 'Unknown')

            reason = self._explain_stock_selection(sym, weight, sector, stock_data_dict)
            stock_reasons.append(reason)

        # Risk explanation
        risk_explanation = self._explain_portfolio_risk(metrics, profile)

        explanation = {
            'profile': profile,
            'strategy': strategy,
            'risk_explanation': risk_explanation,
            'stock_reasons': stock_reasons,
            'key_metrics_summary': self._summarize_metrics(metrics),
        }

        return explanation

    def _explain_stock_selection(self, symbol, weight, sector, stock_data_dict):
        """Explain why a specific stock was selected for the portfolio."""
        df = stock_data_dict.get(symbol)

        reasons = []
        if df is not None and len(df) >= 252:
            close = df['close']
            recent_price = close.iloc[-1]
            year_ago_price = close.iloc[-252] if len(close) >= 252 else close.iloc[0]
            one_year_return = (recent_price / year_ago_price - 1) * 100

            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100

            if one_year_return > 15:
                reasons.append(f"strong 1-year return of {one_year_return:.1f}%")
            elif one_year_return > 0:
                reasons.append(f"positive 1-year return of {one_year_return:.1f}%")
            else:
                reasons.append(f"selected for diversification benefits")

            if volatility < 20:
                reasons.append("low volatility profile")
            elif volatility < 35:
                reasons.append("moderate volatility")

            reasons.append(f"provides exposure to the {sector} sector")
        else:
            reasons.append(f"represents the {sector} sector in the portfolio")

        reason_text = f"**{symbol}** ({weight}% allocation): " + "; ".join(reasons[:3]) + "."
        return reason_text

    def _explain_portfolio_risk(self, metrics, profile):
        """Explain portfolio risk metrics in plain language."""
        if not metrics:
            return "Risk metrics are not available for this portfolio."

        vol = metrics.get('Expected Volatility', 'N/A')
        sharpe = metrics.get('Sharpe Ratio', 'N/A')
        max_dd = metrics.get('Maximum Drawdown', 'N/A')

        explanation = "**Risk Assessment:** "

        # Volatility
        try:
            vol_val = float(vol.replace('%', ''))
            if vol_val < 15:
                explanation += f"Portfolio volatility of {vol} is low, "
            elif vol_val < 25:
                explanation += f"Portfolio volatility of {vol} is moderate, "
            else:
                explanation += f"Portfolio volatility of {vol} is elevated, "
        except Exception:
            explanation += f"Portfolio volatility is {vol}, "

        # Sharpe
        try:
            sharpe_val = float(sharpe)
            if sharpe_val > 1.0:
                explanation += f"and the Sharpe Ratio of {sharpe} indicates good risk-adjusted returns. "
            elif sharpe_val > 0.5:
                explanation += f"and the Sharpe Ratio of {sharpe} indicates adequate compensation for risk. "
            else:
                explanation += f"and the Sharpe Ratio of {sharpe} suggests returns may not fully compensate for risk taken. "
        except Exception:
            explanation += f"with a Sharpe Ratio of {sharpe}. "

        # Max Drawdown
        try:
            dd_val = float(max_dd.replace('%', ''))
            if dd_val < 20:
                explanation += f"The maximum historical drawdown of {max_dd} is manageable."
            elif dd_val < 40:
                explanation += f"The maximum historical drawdown of {max_dd} requires moderate risk tolerance."
            else:
                explanation += f"Investors should be prepared for drawdowns up to {max_dd}."
        except Exception:
            pass

        return explanation

    def _summarize_metrics(self, metrics):
        """Create a brief metrics summary."""
        if not metrics:
            return {}
        return {
            'Return': metrics.get('Expected Annual Return', 'N/A'),
            'Volatility': metrics.get('Expected Volatility', 'N/A'),
            'Sharpe': metrics.get('Sharpe Ratio', 'N/A'),
            'Max Drawdown': metrics.get('Maximum Drawdown', 'N/A'),
        }

    # ================================================================
    # Risk Explanation
    # ================================================================

    def explain_risk_report(self, risk_report):
        """
        Generate a plain-English explanation of a risk assessment report.
        """
        if not risk_report or risk_report.get('volatility_annualized') == 0:
            return "Insufficient data to generate risk explanation."

        symbol = risk_report.get('symbol', 'this stock')
        category = risk_report.get('risk_category', 'Unknown')
        vol = risk_report.get('volatility_annualized', 0)
        sharpe = risk_report.get('sharpe_ratio', 0)
        max_dd = risk_report.get('max_drawdown_pct', 0)
        var_95 = risk_report.get('var_95_daily', 0)

        explanation = f"### Risk Analysis for {symbol}\n\n"
        explanation += f"**Overall Risk Level: {category}**\n\n"

        # Volatility interpretation
        explanation += f"**Volatility ({vol}% annualized):** "
        if vol < 15:
            explanation += (
                f"The stock shows relatively stable price movements. "
                f"For context, an investor can expect daily price swings of roughly "
                f"±{vol/np.sqrt(252):.1f}%. This is typical of large, well-established companies.\n\n"
            )
        elif vol < 25:
            explanation += (
                f"The stock exhibits moderate price fluctuations. "
                f"Daily price changes of ±{vol/np.sqrt(252):.1f}% are common. "
                f"This level of volatility is typical for most NIFTY-50 constituents.\n\n"
            )
        elif vol < 35:
            explanation += (
                f"The stock shows above-average volatility. Investors should expect "
                f"frequent price swings and should have adequate risk tolerance.\n\n"
            )
        else:
            explanation += (
                f"The stock is highly volatile. Price can change significantly in either direction. "
                f"Suitable only for investors with high risk tolerance and a long-term horizon.\n\n"
            )

        # VaR interpretation
        explanation += f"**Value at Risk (95% confidence):** On any given day, there is a 95% chance "
        explanation += f"that losses will not exceed {var_95}%. In other words, losses greater than "
        explanation += f"{var_95}% in a single day are expected only about once a month.\n\n"

        # Sharpe Ratio
        explanation += f"**Sharpe Ratio ({sharpe}):** "
        if sharpe > 1.0:
            explanation += "The stock has delivered good returns relative to the risk taken.\n\n"
        elif sharpe > 0.5:
            explanation += "The stock provides adequate compensation for its risk level.\n\n"
        elif sharpe > 0:
            explanation += "Returns have been modest compared to the risk involved.\n\n"
        else:
            explanation += "The stock has underperformed a risk-free investment on a risk-adjusted basis.\n\n"

        # Max Drawdown
        explanation += f"**Maximum Drawdown ({max_dd}%):** "
        if max_dd < 20:
            explanation += "The worst peak-to-trough decline has been relatively mild.\n\n"
        elif max_dd < 40:
            explanation += "The stock has experienced significant declines during market stress periods.\n\n"
        else:
            explanation += "The stock has suffered severe declines historically, indicating high downside risk.\n\n"

        return explanation

    # ================================================================
    # SHAP-style Feature Attribution (model-agnostic)
    # ================================================================

    def compute_feature_attribution(self, model, X_sample, feature_names, baseline=None):
        """
        Simple model-agnostic feature attribution using perturbation.
        Measures how much each feature contributes to the prediction
        by comparing predictions with and without the feature.
        
        Note: For production use, consider using the SHAP library directly.
        This is a simplified approach for demonstration.
        """
        if baseline is None:
            baseline = np.zeros(X_sample.shape[1])

        try:
            base_pred = model.predict(baseline.reshape(1, -1))[0]
        except Exception:
            return []

        attributions = []
        for i, feature_name in enumerate(feature_names):
            # Perturb: set feature i to baseline value
            X_perturbed = X_sample.copy()
            X_perturbed[0, i] = baseline[i]

            try:
                perturbed_pred = model.predict(X_perturbed)[0]
                attribution = base_pred - perturbed_pred
                attributions.append((feature_name, attribution))
            except Exception:
                attributions.append((feature_name, 0))

        # Sort by absolute attribution
        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return attributions

    # ================================================================
    # Generate Full Explanation Report
    # ================================================================

    def generate_full_report(self, prediction, risk_report, portfolio=None):
        """
        Generate a comprehensive explanation report combining all analyses.
        """
        report = "# Investment Intelligence Report\n\n"

        if prediction:
            report += "## Price Prediction\n\n"
            report += prediction.get('summary', 'No prediction available.') + "\n\n"

            if 'top_factors' in prediction:
                report += "### Key Factors Influencing This Prediction\n\n"
                for factor in prediction['top_factors']:
                    report += f"- **{factor['feature']}** ({factor['importance']}% importance): "
                    report += f"{factor['description']}\n"
                report += "\n"

        if risk_report:
            report += "## Risk Assessment\n\n"
            report += self.explain_risk_report(risk_report)
            report += "\n"

        if portfolio:
            report += "## Portfolio Recommendation\n\n"
            report += f"**Profile:** {portfolio.get('profile', 'Custom')}\n\n"
            report += portfolio.get('strategy', '') + "\n\n"
            report += "### Allocation Reasoning\n\n"
            for reason in portfolio.get('stock_reasons', []):
                report += f"- {reason}\n"
            report += "\n"
            report += portfolio.get('risk_explanation', '') + "\n\n"

        report += "---\n"
        report += "*These explanations are generated automatically to help investors understand "
        report += "the reasoning behind predictions and recommendations. All investment decisions "
        report += "should be made considering personal financial circumstances and risk tolerance.*\n"

        return report
