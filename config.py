"""
Configuration file for the NIFTY-50 Investment Intelligence Platform.
Contains all project-level constants, paths, and parameter settings.
"""

import os

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "nifty_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Dataset URLs
NIFTY50_DATASET_URL = "https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data"
SUPPLEMENTARY_DATASET_URL = "https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020"

# ============================================================
# Stock Prediction Settings
# ============================================================
PREDICTION_HORIZON = 30          # Number of days ahead to forecast
TRAIN_TEST_SPLIT_RATIO = 0.8    # 80% training, 20% testing
LOOKBACK_WINDOW = 60            # Number of past days to use for prediction
VALIDATION_SPLIT = 0.1          # Validation split from training data
RANDOM_SEED = 42

# Model hyperparameters
LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 7
XGBOOST_LEARNING_RATE = 0.05

# ============================================================
# Portfolio Construction Settings
# ============================================================
RISK_FREE_RATE = 0.065           # 6.5% - approximate Indian 10Y bond yield
PORTFOLIO_TOP_N = 12             # Top N stocks to consider for portfolio

# Investor profile risk tolerance ranges
CONSERVATIVE_MAX_VOLATILITY = 0.15      # 15% annualized vol max
BALANCED_MAX_VOLATILITY = 0.25          # 25% annualized vol max
AGGRESSIVE_MAX_VOLATILITY = 0.40        # 40% annualized vol max

# ============================================================
# Risk Assessment Settings
# ============================================================
RISK_METRICS = [
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "beta",
    "calmar_ratio",
    "omega_ratio",
]

# ============================================================
# Technical Indicators Parameters
# ============================================================
MA_PERIODS = [5, 10, 20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3

# ============================================================
# Anomaly Detection Settings
# ============================================================
ANOMALY_ZSCORE_THRESHOLD = 3.0
ANOMALY_VOLUME_SURGE_MULTIPLIER = 3.0
ANOMALY_LOOKBACK_DAYS = 252       # One trading year

# ============================================================
# Dashboard / App Settings
# ============================================================
APP_TITLE = "NIFTY-50 Investment Intelligence Platform"
APP_DESCRIPTION = "AI-Powered Decision Support for Data-Driven Investing"
DEFAULT_SECTOR = "All Sectors"
DEFAULT_STOCK = "RELIANCE"

# Plot styling
COLOR_PALETTE = {
    "primary": "#1f77b4",
    "success": "#2ca02c",
    "warning": "#ff7f0e",
    "danger": "#d62728",
    "neutral": "#7f7f7f",
    "background": "#fafafa",
}
