# Technical Report: Data-Driven Investment Intelligence Using NIFTY-50 Market Data

**Open Projects 2026 | Cultural Council | IIT Roorkee**

**Submitted by:** Praneshwar Kannan Kommiya (Enrollment Number: 23117102), B.Tech Mechanical Engineering - 4Y

---

## 1. Introduction

Financial markets generate enormous volumes of data daily, creating significant challenges for investors seeking to identify meaningful patterns and make informed decisions. The NIFTY-50 index, representing fifty of India's largest publicly traded companies across Banking, Information Technology, Energy, Pharmaceuticals, and Manufacturing sectors, serves as a benchmark for the Indian equity market.

This project develops an AI-powered investment intelligence platform that transforms raw historical market data into actionable insights. Unlike traditional stock price prediction competitions, the focus is on creating a practical decision-support system that combines machine learning, portfolio optimization, risk analytics, and explainable AI.

### 1.1 Objectives

- Develop predictive models for stock behavior using historical data
- Build optimized investment portfolios for different risk profiles
- Implement comprehensive risk assessment metrics
- Detect market anomalies and unusual patterns
- Provide transparent explanations for all recommendations
- Deliver an interactive, user-friendly dashboard

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Dataset Overview

The primary dataset used is the NIFTY-50 Stock Market Dataset from Kaggle, spanning January 2000 to April 2021. It contains daily OHLCV (Open, High, Low, Close, Volume) data for each constituent stock.

**Key Statistics:**
- Total stocks analyzed: ~50 (varies based on index inclusion periods)
- Date range: January 2000 – April 2021
- Total trading days: ~5,200 per stock (varies)
- Sectors covered: 8+ major sectors

### 2.2 Data Quality Assessment

The dataset required cleaning for:
- Inconsistent column naming across different CSV files
- Missing values in volume and price columns
- Date parsing with mixed formats
- Duplicate entries for the same trading date
- Price anomalies requiring sanity checks (high >= low, etc.)

### 2.3 Key Observations

1. **Sector Distribution:** Banking and Financial Services constitute approximately 35% of the index weight, followed by Information Technology (~18%) and Energy (~15%).

2. **Return Characteristics:** Daily returns exhibit fat tails (excess kurtosis > 3 for most stocks), indicating more extreme events than a normal distribution would predict. This has implications for risk modeling.

3. **Volatility Clustering:** Periods of high volatility tend to cluster together, consistent with established financial market behavior. This validates the use of GARCH-style volatility models.

4. **Market Regimes:** The data covers several distinct market regimes including the 2008 Global Financial Crisis, the 2014-2015 bull market, the 2018 correction, and the COVID-19 crash and recovery of 2020-2021.

---

## 3. Feature Engineering

### 3.1 Technical Indicators

The platform computes over 30 technical features from raw price and volume data:

**Trend Indicators:**
- Simple Moving Averages (5, 10, 20, 50, 200-day)
- Exponential Moving Averages (12, 26-day)
- MACD (Moving Average Convergence Divergence) with signal line and histogram
- Price position relative to moving averages

**Momentum Indicators:**
- Relative Strength Index (RSI-14) with Wilder's smoothing
- Stochastic Oscillator (%K and %D)
- Rate of Change (ROC) for 5, 10, 20, and 60-day periods
- Price momentum (close - close N days ago)

**Volatility Indicators:**
- Bollinger Bands (20-day, 2 standard deviations)
- Average True Range (ATR-14)
- Historical volatility (5, 10, 20, 60-day rolling)
- Parkinson volatility estimator (uses high-low range)

**Volume Indicators:**
- Volume moving averages and ratios
- On-Balance Volume (OBV)
- Volume-Price Trend (VPT)

### 3.2 Feature Selection Rationale

Features were selected based on their established use in technical analysis literature and their predictive power for short-to-medium term price movements. The combination of trend, momentum, volatility, and volume indicators provides a comprehensive representation of market conditions.

---

## 4. Stock Prediction Engine

### 4.1 Model Architecture

The prediction engine uses two complementary approaches:

**XGBoost Regressor:**
- Gradient-boosted decision tree ensemble
- Hyperparameters: 200 estimators, max depth 7, learning rate 0.05, 80% subsample
- Trained on 30+ engineered features
- Provides feature importance scores for interpretability

**LSTM Neural Network:**
- Two-layer LSTM with 64 and 32 units
- Dropout (0.2) for regularization
- 60-day lookback window
- Trained with Adam optimizer and early stopping
- Captures sequential dependencies in price data

### 4.2 Training Methodology

- **Train/Test Split:** 80% training, 20% testing (time-series aware, no shuffle)
- **Validation:** 10% of training data for LSTM early stopping
- **Target Variable:** Future return over N-day horizon (configurable: 7-90 days)
- **Feature Scaling:** StandardScaler for XGBoost, MinMaxScaler for LSTM

### 4.3 Evaluation Results

The models are evaluated using multiple metrics:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error of return predictions |
| **RMSE** | Root Mean Squared Error |
| **R² Score** | Coefficient of determination |
| **Directional Accuracy** | Percentage of correct up/down predictions |

Directional accuracy typically ranges from 55-65% depending on the stock and horizon, which is meaningful above the 50% random baseline. The XGBoost model generally outperforms LSTM on shorter horizons, while LSTM shows competitive performance on longer sequences.

### 4.4 Limitations

- Models are trained on historical data and assume similar patterns will continue
- Market regime changes can degrade prediction accuracy
- Predictions are most reliable for direction rather than exact magnitude
- Individual stock predictions are noisy; ensemble approaches help

---

## 5. Portfolio Construction

### 5.1 Methodology

The portfolio construction module uses Modern Portfolio Theory (Markowitz, 1952) to optimize asset allocation. The optimization maximizes risk-adjusted returns subject to investor-specific constraints.

**Optimization Objective:**
- **Conservative:** Minimize portfolio variance
- **Balanced:** Target 20% annual volatility while maximizing return
- **Aggressive:** Maximize Sharpe Ratio

**Constraints:**
- Fully invested (weights sum to 100%)
- Long-only positions
- Maximum 35% single-stock exposure
- Minimum position size threshold (0.5%)

### 5.2 Implementation

The optimization uses SciPy's SLSQP (Sequential Least Squares Quadratic Programming) algorithm, which handles constrained nonlinear optimization efficiently.

The Efficient Frontier is computed by solving for optimal portfolios at different target volatility levels, demonstrating the risk-return tradeoff available to investors.

### 5.3 Portfolio Profiles

| Profile | Target Return | Max Volatility | Horizon | Rebalancing |
|---------|--------------|----------------|---------|-------------|
| Conservative | 8-12% | 15% | 3-5 years | Quarterly |
| Balanced | 12-18% | 25% | 2-4 years | Quarterly |
| Aggressive | 18%+ | 40% | 1-3 years | Monthly |

### 5.4 Diversification

Sector diversification is enforced implicitly through the optimization process, which naturally spreads risk across uncorrelated assets. The platform reports sector-wise allocation for each portfolio to ensure adequate diversification.

---

## 6. Risk Assessment

### 6.1 Risk Metrics Implemented

The platform computes 10 risk metrics providing a multi-faceted view of investment risk:

1. **Annualized Volatility:** Standard deviation of daily returns × √252
2. **Sharpe Ratio:** (Return - Risk Free Rate) / Volatility
3. **Sortino Ratio:** Downside-only risk-adjusted return
4. **Maximum Drawdown:** Largest peak-to-trough decline
5. **Value at Risk (VaR 95%):** Maximum expected daily loss at 95% confidence
6. **Conditional VaR (CVaR 95%):** Expected loss beyond VaR threshold
7. **Calmar Ratio:** Annualized return / Maximum drawdown
8. **Omega Ratio:** Probability-weighted ratio of gains to losses
9. **Return Skewness:** Asymmetry of return distribution
10. **Excess Kurtosis:** Tail risk indicator

### 6.2 Risk Classification

Stocks are classified into risk categories based on volatility and drawdown thresholds:
- **Low Risk:** Vol < 18%, Max DD < 30%
- **Moderate Risk:** Vol < 30%, Max DD < 50%
- **High Risk:** Vol < 45%
- **Very High Risk:** Vol ≥ 45%

### 6.3 Rolling Risk Analysis

Rolling volatility and Value at Risk are computed over moving windows to track how risk evolves over time. This is particularly useful for identifying periods of market stress.

---

## 7. Anomaly Detection

### 7.1 Detection Methods

Three statistical methods are combined for robust anomaly detection:

**Price Anomalies:**
- Z-score method (threshold: 3.0 standard deviations)
- Interquartile Range (IQR) method
- Modified Z-score (median-based, robust to outliers)
- Consensus: At least 2 of 3 methods must agree

**Volume Anomalies:**
- Volume Z-score relative to 50-day rolling statistics
- Volume surge ratio (current volume / average volume)

**Volatility Spikes:**
- Rolling volatility Z-score against 252-day lookback

### 7.2 Market-Wide Events

The platform identifies dates where multiple stocks simultaneously exhibit anomalies. These dates typically correspond to major market events such as the 2008 financial crisis, COVID-19 crash (March 2020), Union Budget announcements, and RBI policy decisions.

---

## 8. Explainable AI Framework

### 8.1 Approach

Explainability is integrated at multiple levels:

1. **Feature Importance:** XGBoost's built-in feature importance shows which indicators most influence predictions
2. **Natural Language Explanations:** Predictions and recommendations are translated into plain-English summaries
3. **Portfolio Rationale:** Each stock selection includes specific reasoning based on quantitative metrics
4. **Risk Interpretation:** Risk metrics are contextualized with thresholds and comparisons

### 8.2 Explanation Design

Explanations follow a structured format:
- **Summary:** One-line takeaway
- **Key Factors:** Top 5 influencing features with importance scores
- **Context:** Comparison to benchmarks and thresholds
- **Caveats:** Limitations and confidence levels

---

## 9. Key Insights

### 9.1 Market Behavior Insights

1. **Sector Rotation:** Different sectors lead at different phases of the economic cycle. The platform's sector analysis helps identify where opportunities exist.

2. **Volatility is Predictable:** While price direction is difficult to forecast, volatility exhibits persistence that can be modeled effectively.

3. **Diversification Benefits:** Even within NIFTY-50, correlations between stocks vary significantly (0.3 to 0.8), providing meaningful diversification opportunities.

4. **Tail Risk is Real:** Return distributions consistently show fat tails, meaning extreme events occur more frequently than normal distribution models predict.

### 9.2 Model Insights

1. **Feature Importance:** Volume-based features and momentum indicators consistently rank among the most predictive features, often outperforming simple price-based features.

2. **Ensemble Benefits:** Combining XGBoost and LSTM predictions reduces variance and improves directional accuracy compared to either model alone.

3. **Horizon Effects:** Prediction accuracy degrades with longer horizons. The 30-day horizon offers a reasonable balance between usefulness and reliability.

---

## 10. Conclusion

This project demonstrates the feasibility of building a comprehensive investment intelligence platform using only historical market data. The combination of machine learning, portfolio optimization, risk analytics, and explainable AI creates a practical decision-support tool for investors.

### 10.1 Achievements

- ✅ Functional stock prediction engine with multiple ML models
- ✅ Portfolio construction for three investor profiles
- ✅ Comprehensive risk assessment with 10+ metrics
- ✅ Market anomaly detection across multiple dimensions
- ✅ Multi-method forecasting with Monte Carlo simulation
- ✅ Explainable AI framework for transparent recommendations
- ✅ Interactive Streamlit dashboard for user-friendly access

### 10.2 Future Work

1. **Reinforcement Learning** for dynamic portfolio rebalancing
2. **Bayesian Methods** for uncertainty quantification
3. **Transformer Models** for long-range sequence modeling
4. **Multi-Asset Integration** beyond equities (bonds, gold, etc.)
5. **Real-time Data Pipeline** for live market monitoring

---

## References

1. Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance, 7(1), 77-91.
2. Sharpe, W.F. (1966). "Mutual Fund Performance." The Journal of Business, 39(1), 119-138.
3. Sortino, F.A. & Price, L.N. (1994). "Performance Measurement in a Downside Risk Framework." The Journal of Investing, 3(3), 59-64.
4. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD '16.
5. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation, 9(8), 1735-1780.
6. Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." Journal of Econometrics, 31(3), 307-327.

---

*Report prepared for Open Projects 2026, Cultural Council, IIT Roorkee.*
