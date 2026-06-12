# NIFTY-50 Investment Intelligence Platform

**Data-Driven Decision Support for Stock Market Investing**

---

## Project Information

| Field | Details |
|-------|---------|
| **Name** | Praneshwar Kannan Kommiya |
| **Enrollment Number** | 23117102 |
| **Program** | B.Tech Mechanical Engineering (4th Year) |
| **Institution** | Indian Institute of Technology, Roorkee |
| **Event** | Open Projects 2026, Cultural Council |
| **Problem Statement** | Data-Driven Investment Intelligence Using NIFTY-50 Market Data |

---

## About The Project

An AI-powered investment intelligence platform that transforms raw NIFTY-50 historical market data into actionable insights for investors. The platform combines machine learning, statistical modeling, and financial analytics to provide:

- **Stock Predictions** using XGBoost and LSTM models
- **Portfolio Construction** for three investor risk profiles
- **Comprehensive Risk Assessment** with 10+ metrics
- **Market Anomaly Detection** for unusual market patterns
- **Multi-Method Forecasting** including Monte Carlo simulation
- **Explainable AI** providing transparent reasoning

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/palaemon-codes/nifty50-investment-intelligence.git
cd nifty50-investment-intelligence
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you face issues with TensorFlow installation, you can run the platform without it. The LSTM model will simply be skipped, and XGBoost predictions will still work.

### 4. Download the Dataset

Download the NIFTY-50 dataset from Kaggle:
- **Primary Dataset:** [NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
- **Supplementary:** [India Stock Data NSE 1990-2020](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020)

Extract all CSV files into the `data/` folder:

```
nifty50-investment-intelligence/
├── data/
│   ├── RELIANCE.csv
│   ├── TCS.csv
│   ├── HDFCBANK.csv
│   ├── INFY.csv
│   └── ... (all other stock CSVs)
├── src/
├── app.py
└── ...
```

### 5. Launch the Platform

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Project Structure

```
nifty50-investment-intelligence/
│
├── app.py                      # Main Streamlit dashboard application
├── config.py                   # Project configuration and parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading, cleaning, preprocessing
│   ├── feature_engineering.py  # Technical indicators computation
│   ├── stock_predictor.py      # ML models (XGBoost, LSTM)
│   ├── portfolio.py            # Portfolio optimization & construction
│   ├── risk_assessment.py      # Risk metrics & analysis
│   ├── anomaly_detection.py    # Market anomaly detection
│   ├── explainable_ai.py       # XAI explanations framework
│   └── forecasting.py          # Multi-method forecasting
│
├── data/                       # Dataset directory (CSV files)
├── notebooks/                  # Jupyter notebooks (EDA)
└── reports/                    # Generated reports
```

---

## Features & Modules

### Stock Analysis
- Interactive candlestick charts with Bollinger Bands & Moving Averages
- Technical indicators: RSI, MACD, ATR, Stochastic Oscillator
- ML-powered price prediction with directional accuracy
- Feature importance visualization for model interpretability

### Portfolio Builder
- **Conservative Portfolio:** Capital preservation, low volatility, stable returns
- **Balanced Portfolio:** Growth + stability, sector diversification
- **Aggressive Portfolio:** Maximum growth potential, higher risk tolerance
- Efficient Frontier visualization
- Sector-wise allocation breakdown
- Stock selection rationale for each recommendation

### Risk Assessment
- Annualized Volatility
- Sharpe Ratio & Sortino Ratio
- Value at Risk (VaR) & Conditional VaR
- Maximum Drawdown analysis
- Calmar Ratio & Omega Ratio
- Rolling risk metrics
- Cross-stock risk comparison

### Anomaly Detection
- Price anomalies (extreme daily movements)
- Volume surge detection
- Volatility spike identification
- Market-wide event detection
- Regime change analysis

### Forecasting
- Linear trend regression with confidence bands
- Holt's Exponential Smoothing
- Monte Carlo simulation (500+ paths)
- Volatility forecasting (GARCH-style)
- Ensemble forecast combining multiple methods

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Web Framework** | Streamlit |
| **Visualization** | Plotly, Matplotlib |
| **ML Models** | XGBoost, TensorFlow/Keras (LSTM) |
| **Optimization** | SciPy (SLSQP) |
| **Data Processing** | Pandas, NumPy |
| **Statistics** | SciPy, Scikit-learn |

---

## Machine Learning Approach

### Stock Prediction Engine
- **XGBoost Regressor:** Gradient boosting on 30+ technical features
- **LSTM Neural Network:** Deep learning for sequential price patterns
- **Evaluation:** MAE, RMSE, R² Score, Directional Accuracy

### Portfolio Optimization
- **Modern Portfolio Theory (MPT)** framework
- **Efficient Frontier** computation
- **Risk-based allocation** using SciPy SLSQP optimizer
- **Constraints:** Max 35% per stock, fully invested, long-only

### Forecasting Methods
- Linear trend regression with 95% confidence intervals
- Holt's exponential smoothing with trend component
- Geometric Brownian Motion Monte Carlo (500-1000 simulations)
- EWMA/GARCH-style volatility forecasting

---

## Reproducing Results

1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Download and place the dataset CSV files in the `data/` folder
3. Run `streamlit run app.py`
4. Use the sidebar to load data and navigate to desired analysis modules
5. All computations are performed on-demand with the loaded data

---

## Disclaimer

This platform is built for **educational purposes** as part of Open Projects 2026 at IIT Roorkee. It does NOT constitute financial advice. All predictions, portfolio recommendations, and risk assessments are based on historical data and statistical models. Past performance does not guarantee future results.

Investment decisions should be made considering personal financial circumstances, risk tolerance, and investment goals. Please consult a qualified financial advisor before making investment decisions.

---

## License

This project is submitted as part of the Open Projects 2026 competition organized by the Cultural Council, IIT Roorkee.
