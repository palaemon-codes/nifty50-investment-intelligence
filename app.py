"""
NIFTY-50 Investment Intelligence Platform
Main Streamlit Dashboard Application

Built by: Praneshwar Kannan Kommiya (23117102)
B.Tech Mechanical Engineering - 4Y, IIT Roorkee
Open Projects 2026, Cultural Council

A comprehensive AI-powered platform for data-driven investment decision support.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import *
from data_loader import NIFTY50DataLoader
from feature_engineering import FeatureEngineer
from stock_predictor import StockPredictionEngine
from portfolio import PortfolioConstructor
from risk_assessment import RiskAssessor
from anomaly_detection import AnomalyDetector
from explainable_ai import ExplainableAI
from forecasting import Forecaster

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS for better appearance
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #16213e;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f3460;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.85rem;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }
    /* Make dataframe look better */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Initialization
# ============================================================
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictionEngine()
if 'risk_assessor' not in st.session_state:
    st.session_state.risk_assessor = RiskAssessor()
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()
if 'explainer' not in st.session_state:
    st.session_state.explainer = ExplainableAI()
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = Forecaster()
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None


# ============================================================
# Helper Functions
# ============================================================

def load_data_from_folder(folder_path):
    """Load data from a folder of CSV files."""
    with st.spinner("Loading NIFTY-50 stock data... This may take a moment."):
        loader = NIFTY50DataLoader(folder_path)
        loader.load_company_metadata()
        success = loader.load_from_folder(folder_path)
        if success:
            st.session_state.data_loader = loader
            st.session_state.data_loaded = True
            return True
    return False


def create_candlestick_chart(df, symbol, show_volume=True, show_ma=True):
    """Create an interactive candlestick chart with Plotly."""
    if df is None or len(df) < 5:
        return go.Figure()

    # Use last 252 trading days (~1 year) for clarity
    plot_df = df.tail(252).copy()

    fig = make_subplots(
        rows=2 if show_volume else 1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3] if show_volume else [1.0],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=plot_df['date'],
            open=plot_df['open'],
            high=plot_df['high'],
            low=plot_df['low'],
            close=plot_df['close'],
            name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )

    # Moving averages
    if show_ma:
        if 'sma_20' in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df['date'], y=plot_df['sma_20'],
                           mode='lines', name='SMA 20',
                           line=dict(color='orange', width=1)),
                row=1, col=1,
            )
        if 'sma_50' in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df['date'], y=plot_df['sma_50'],
                           mode='lines', name='SMA 50',
                           line=dict(color='blue', width=1)),
                row=1, col=1,
            )

    # Bollinger Bands
    if 'bb_upper' in plot_df.columns and 'bb_lower' in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df['date'], y=plot_df['bb_upper'],
                       mode='lines', name='BB Upper',
                       line=dict(color='gray', width=0.5, dash='dash'),
                       showlegend=True),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=plot_df['date'], y=plot_df['bb_lower'],
                       mode='lines', name='BB Lower',
                       line=dict(color='gray', width=0.5, dash='dash'),
                       fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                       showlegend=True),
            row=1, col=1,
        )

    # Volume bars
    if show_volume and 'volume' in plot_df.columns:
        colors = ['#26a69a' if plot_df['close'].iloc[i] >= plot_df['open'].iloc[i]
                  else '#ef5350' for i in range(len(plot_df))]
        fig.add_trace(
            go.Bar(x=plot_df['date'], y=plot_df['volume'],
                   name='Volume', marker_color=colors,
                   showlegend=False),
            row=2, col=1,
        )

    # Layout
    fig.update_layout(
        title=f"{symbol} - Price Chart",
        yaxis_title="Price (Rs.)",
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_indicator_chart(df, indicator_col, title, color='blue', threshold_lines=None):
    """Create a simple line chart for a technical indicator."""
    if df is None or indicator_col not in df.columns:
        return go.Figure()

    plot_df = df.tail(252).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df[indicator_col],
                   mode='lines', name=indicator_col.replace('_', ' ').title(),
                   line=dict(color=color, width=2))
    )

    # Add threshold lines if provided
    if threshold_lines:
        for value, label, line_color in threshold_lines:
            fig.add_hline(y=value, line_dash="dash", line_color=line_color,
                          annotation_text=label)

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=350,
        hovermode='x unified',
    )

    return fig


def display_metric_card(value, label, color_scheme="primary"):
    """Display a styled metric card."""
    colors = {
        "primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "success": "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        "warning": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        "info": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    }
    bg = colors.get(color_scheme, colors["primary"])
    st.markdown(f"""
    <div style="background: {bg}; padding: 1rem; border-radius: 12px; color: white; text-align: center; margin: 0.3rem 0;">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
    st.markdown("## 📊 Navigation")

    page = st.radio(
        "Select Module",
        ["🏠 Home", "📈 Stock Analysis", "💼 Portfolio Builder",
         "⚠️ Risk Assessment", "🔍 Anomaly Detection", "🔮 Forecasting",
         "📋 About & Guide"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Data loading section in sidebar
    st.markdown("### 📂 Data Management")

    # Auto-detect data folder
    data_folder = DATA_DIR
    if not os.path.exists(data_folder):
        # fallback: try inside the project too
        alt_path = os.path.join(os.path.dirname(__file__), "data", "nifty_data")
        if os.path.exists(alt_path):
            data_folder = alt_path

    st.caption(f"Looking in: `{data_folder}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Load Data", use_container_width=True):
            load_data_from_folder(data_folder)

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.data_loaded = False
            st.session_state.data_loader = None
            st.rerun()

    if st.session_state.data_loaded:
        loader = st.session_state.data_loader
        st.success(f"✅ {len(loader.all_symbols)} stocks loaded")
    else:
        st.info("📁 Place the nifty_data folder in ~/Downloads/ and click Load Data")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #888;">
    <strong>Built by:</strong><br>
    Praneshwar Kannan Kommiya<br>
    23117102 | B.Tech ME - 4Y<br>
    Open Projects 2026<br>
    Cultural Council, IIT Roorkee
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: Home
# ============================================================
if page == "🏠 Home":
    st.markdown('<div class="main-header">📊 NIFTY-50 Investment Intelligence Platform</div>',
                unsafe_allow_html=True)
    st.markdown("##### AI-Powered Decision Support for Data-Driven Investing")

    st.markdown("---")

    # Platform overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Welcome to the Investment Intelligence Platform

        This platform transforms raw NIFTY-50 historical market data into **actionable investment insights**
        using machine learning, statistical modeling, and financial analytics.

        #### What You Can Do Here:

        - **📈 Analyze Stocks** — Deep-dive into individual stocks with technical indicators,
          price charts, and ML-powered predictions
        - **💼 Build Portfolios** — Generate optimized portfolios for Conservative, Balanced,
          and Aggressive investor profiles
        - **⚠️ Assess Risk** — Evaluate risk metrics including volatility, Sharpe Ratio,
          Value at Risk, and maximum drawdown
        - **🔍 Detect Anomalies** — Identify unusual market patterns, volume surges,
          and volatility spikes
        - **🔮 Forecast Trends** — Multi-method forecasting using regression, exponential
          smoothing, and Monte Carlo simulation

        #### Dataset:
        The platform uses the **NIFTY-50 Stock Market Dataset** spanning from January 2000
        to April 2021, covering 50 major Indian companies across Banking, IT, Energy,
        Pharma, FMCG, Auto, and Manufacturing sectors.
        """)

        st.markdown('<div class="info-box">'
                     '💡 <strong>Getting Started:</strong> First, load the dataset using the sidebar. '
                     'Then navigate to any module to start your analysis.'
                     '</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 Platform Stats")
        if st.session_state.data_loaded:
            loader = st.session_state.data_loader
            n_stocks = len(loader.all_symbols)
            n_sectors = len(loader.get_all_sectors())

            display_metric_card(str(n_stocks), "Stocks Available", "primary")
            display_metric_card(str(n_sectors), "Sectors Covered", "success")

            # Show sector breakdown
            sectors = loader.get_all_sectors()
            sector_counts = {s: len(loader.get_stocks_by_sector(s)) for s in sectors}
            if sector_counts:
                fig = px.pie(
                    values=list(sector_counts.values()),
                    names=list(sector_counts.keys()),
                    title="Sector Distribution",
                    hole=0.4,
                )
                fig.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load data to see platform statistics")

    st.markdown("---")

    # Quick features preview
    st.markdown("### 🚀 Platform Features at a Glance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🤖</h2>
            <strong>ML Predictions</strong>
            <p style="font-size: 0.85rem;">XGBoost & LSTM models for price direction and return forecasting</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>📐</h2>
            <strong>Portfolio Optimization</strong>
            <p style="font-size: 0.85rem;">Modern Portfolio Theory with Efficient Frontier analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🛡️</h2>
            <strong>Risk Analytics</strong>
            <p style="font-size: 0.85rem;">10+ risk metrics including VaR, CVaR, and drawdown analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🔮</h2>
            <strong>Forecasting</strong>
            <p style="font-size: 0.85rem;">Monte Carlo simulation & multi-method trend forecasting</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: Stock Analysis
# ============================================================
elif page == "📈 Stock Analysis":
    st.markdown('<div class="main-header">📈 Stock Analysis & Prediction</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first using the sidebar.")
        st.stop()

    loader = st.session_state.data_loader

    # Stock selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            loader.all_symbols,
            index=loader.all_symbols.index("RELIANCE") if "RELIANCE" in loader.all_symbols else 0,
        )
    with col2:
        horizon = st.selectbox("Prediction Horizon", [7, 15, 30, 60, 90], index=2)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("🔍 Run Analysis", use_container_width=True)

    if run_analysis:
        with st.spinner(f"Analyzing {selected_stock}... This may take a minute."):
            df = loader.get_stock(selected_stock)
            if df is None:
                st.error(f"No data available for {selected_stock}")
                st.stop()

            # Feature engineering
            fe = st.session_state.feature_engineer
            df_featured = fe.compute_all_features(df)

            # Stock info
            sector = loader.get_sector(selected_stock)
            latest_price = df['close'].iloc[-1]

            st.markdown("---")

            # Key metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                display_metric_card(f"₹{latest_price:.2f}", "Current Price", "primary")
            with col2:
                change_1m = (df['close'].iloc[-1] / df['close'].iloc[-22] - 1) * 100 if len(df) >= 22 else 0
                color = "success" if change_1m >= 0 else "warning"
                display_metric_card(f"{change_1m:+.2f}%", "1-Month Change", color)
            with col3:
                change_1y = (df['close'].iloc[-1] / df['close'].iloc[-252] - 1) * 100 if len(df) >= 252 else 0
                color = "success" if change_1y >= 0 else "warning"
                display_metric_card(f"{change_1y:+.2f}%", "1-Year Change", color)
            with col4:
                display_metric_card(sector, "Sector", "info")
            with col5:
                n_days = len(df)
                display_metric_card(str(n_days), "Trading Days", "info")

            st.markdown("---")

            # ---- Tab layout for stock analysis ----
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Price & Indicators", "🤖 ML Prediction", "⚠️ Risk Profile", "📝 Summary"]
            )

            # Tab 1: Price chart and technical indicators
            with tab1:
                st.markdown("### Candlestick Chart with Technical Indicators")
                fig = create_candlestick_chart(df_featured, selected_stock, show_volume=True)
                st.plotly_chart(fig, use_container_width=True)

                # Technical indicators
                st.markdown("### Technical Indicators")
                col1, col2 = st.columns(2)

                with col1:
                    if 'rsi' in df_featured.columns:
                        fig_rsi = create_indicator_chart(
                            df_featured, 'rsi', 'RSI (14)',
                            color='purple',
                            threshold_lines=[(70, 'Overbought', 'red'), (30, 'Oversold', 'green')]
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)

                    if 'macd' in df_featured.columns:
                        fig_macd = go.Figure()
                        plot_df = df_featured.tail(252)
                        fig_macd.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['macd'],
                                                       mode='lines', name='MACD',
                                                       line=dict(color='blue')))
                        fig_macd.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['macd_signal'],
                                                       mode='lines', name='Signal',
                                                       line=dict(color='red')))
                        fig_macd.add_trace(go.Bar(x=plot_df['date'], y=plot_df['macd_histogram'],
                                                   name='Histogram',
                                                   marker_color=np.where(plot_df['macd_histogram'] >= 0, 'green', 'red')))
                        fig_macd.update_layout(title='MACD', template='plotly_white', height=350)
                        st.plotly_chart(fig_macd, use_container_width=True)

                with col2:
                    if 'bb_upper' in df_featured.columns:
                        fig_bb = go.Figure()
                        plot_df = df_featured.tail(252)
                        fig_bb.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['close'],
                                                     mode='lines', name='Close',
                                                     line=dict(color='black')))
                        fig_bb.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['bb_upper'],
                                                     mode='lines', name='BB Upper',
                                                     line=dict(color='gray', dash='dash')))
                        fig_bb.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['bb_lower'],
                                                     mode='lines', name='BB Lower',
                                                     line=dict(color='gray', dash='dash')))
                        fig_bb.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['bb_middle'],
                                                     mode='lines', name='BB Middle',
                                                     line=dict(color='orange', dash='dot')))
                        fig_bb.update_layout(title='Bollinger Bands (20,2)', template='plotly_white', height=350)
                        st.plotly_chart(fig_bb, use_container_width=True)

                    if 'atr' in df_featured.columns:
                        fig_atr = create_indicator_chart(
                            df_featured, 'atr_percent', 'Average True Range (%)', color='brown'
                        )
                        st.plotly_chart(fig_atr, use_container_width=True)

            # Tab 2: ML Predictions
            with tab2:
                st.markdown("### 🤖 Machine Learning Price Prediction")

                try:
                    predictor = st.session_state.predictor
                    results = predictor.train_for_stock(df_featured, selected_stock, horizon)

                    if results:
                        st.markdown("#### Model Performance Metrics")
                        metrics_df = pd.DataFrame(list(results.values()))
                        st.dataframe(metrics_df, use_container_width=True)

                        # Direction accuracy gauge
                        if 'XGBoost' in results:
                            dir_acc = results['XGBoost']['Directional_Accuracy']
                            st.markdown(f"#### Directional Accuracy: {dir_acc}%")
                            st.progress(dir_acc / 100,
                                        text=f"The model correctly predicts price direction {dir_acc}% of the time")

                        # Feature importance
                        st.markdown("#### Top Features Influencing Prediction")
                        importance = predictor.get_model_interpretation()
                        if importance:
                            imp_df = pd.DataFrame(importance, columns=['Feature', 'Importance'])
                            fig_imp = px.bar(
                                imp_df.head(15),
                                x='Importance', y='Feature',
                                orientation='h',
                                title='Feature Importance (XGBoost)',
                            )
                            fig_imp.update_layout(height=400)
                            st.plotly_chart(fig_imp, use_container_width=True)

                        # Future prediction
                        st.markdown("---")
                        st.markdown(f"### Prediction for Next {horizon} Days")
                        future_pred = predictor.predict_future(df_featured, horizon)

                        if future_pred:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                direction = future_pred['direction']
                                emoji = "🟢" if direction == "UP" else "🔴"
                                display_metric_card(
                                    f"{emoji} {direction}",
                                    "Predicted Direction",
                                    "success" if direction == "UP" else "warning"
                                )
                            with col2:
                                display_metric_card(
                                    f"{future_pred['predicted_return_pct']:+.2f}%",
                                    "Expected Return",
                                    "success" if future_pred['predicted_return_pct'] > 0 else "warning"
                                )
                            with col3:
                                display_metric_card(
                                    f"₹{future_pred['predicted_price_xgboost']:.2f}",
                                    "Predicted Price (XGBoost)",
                                    "info"
                                )

                            # Explanation
                            st.markdown('<div class="info-box">'
                                        f'<strong>What this means:</strong> Based on historical patterns and '
                                        f'technical indicators, {selected_stock} is expected to move '
                                        f'<strong>{direction.lower()}</strong> over the next {horizon} days. '
                                        f'The model predicts a {future_pred["predicted_return_pct"]:+.2f}% return. '
                                        f'This is a statistical estimate and should be used alongside other analysis.'
                                        '</div>', unsafe_allow_html=True)
                    else:
                        st.info("Could not train prediction models. XGBoost may not be installed.")

                except Exception as e:
                    st.warning(f"Prediction module encountered an issue: {e}")
                    st.info("Install required packages: `pip install xgboost tensorflow scikit-learn`")

            # Tab 3: Risk Profile
            with tab3:
                st.markdown("### ⚠️ Stock Risk Assessment")

                risk_assessor = st.session_state.risk_assessor
                risk_report = risk_assessor.assess_stock(df, selected_stock)

                if risk_report:
                    # Risk category badge
                    category = risk_report['risk_category']
                    cat_color = {
                        "Low Risk": "green",
                        "Moderate Risk": "orange",
                        "High Risk": "red",
                        "Very High Risk": "darkred",
                    }.get(category, "gray")

                    st.markdown(f"#### Risk Category: <span style='color:{cat_color};font-weight:bold;'>{category}</span>",
                                unsafe_allow_html=True)

                    # Risk metrics grid
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        display_metric_card(f"{risk_report['volatility_annualized']}%", "Annual Volatility",
                                            "warning" if risk_report['volatility_annualized'] > 25 else "success")
                    with col2:
                        display_metric_card(str(risk_report['sharpe_ratio']), "Sharpe Ratio",
                                            "success" if risk_report['sharpe_ratio'] > 0.5 else "warning")
                    with col3:
                        display_metric_card(f"{risk_report['max_drawdown_pct']}%", "Max Drawdown",
                                            "warning" if risk_report['max_drawdown_pct'] > 30 else "success")
                    with col4:
                        display_metric_card(f"{risk_report['var_95_daily']}%", "Daily VaR (95%)", "info")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        display_metric_card(str(risk_report['sortino_ratio']), "Sortino Ratio", "info")
                    with col2:
                        display_metric_card(str(risk_report['calmar_ratio']), "Calmar Ratio", "info")
                    with col3:
                        display_metric_card(str(risk_report['omega_ratio']), "Omega Ratio", "info")
                    with col4:
                        display_metric_card(f"{risk_report['positive_days_pct']}%", "Positive Days", "info")

                    # Additional stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"- **Total Return:** {risk_report['total_return_pct']}%")
                        st.markdown(f"- **Return Skewness:** {risk_report['return_skewness']}")
                        st.markdown(f"- **CVaR (95%):** {risk_report['cvar_95_daily']}% daily")
                    with col2:
                        st.markdown(f"- **Return Kurtosis:** {risk_report['return_kurtosis']}")
                        st.markdown(f"- **Max DD Duration:** {risk_report['max_drawdown_days']} days")
                        st.markdown(f"- **Data Points:** {risk_report['data_points']}")

                    # Risk interpretation
                    st.markdown("---")
                    st.markdown("### Risk Interpretation")
                    explainer = st.session_state.explainer
                    risk_text = explainer.explain_risk_report(risk_report)
                    st.markdown(risk_text)

            # Tab 4: Summary
            with tab4:
                st.markdown("### 📝 Investment Summary")

                # Generate comprehensive summary
                st.markdown(f"""
                #### {selected_stock} — {sector} Sector

                **Current Price:** ₹{latest_price:.2f}

                **About this stock:**
                {selected_stock} is a constituent of the NIFTY-50 index in the {sector} sector.
                The stock has {len(df)} trading days of historical data available for analysis.

                **Key Takeaways:**
                - The technical indicators and ML models provide a data-driven perspective
                  on potential price movements, but all predictions carry inherent uncertainty.
                - Risk assessment helps quantify the potential downside and volatility
                  you might experience as an investor.
                - Always consider your personal financial goals, investment horizon, and
                  risk tolerance before making investment decisions.

                **Disclaimer:** This analysis is based on historical data and statistical models.
                Past performance does not guarantee future results. The predictions and insights
                provided are for educational and informational purposes only and do not constitute
                financial advice.
                """)


# ============================================================
# PAGE: Portfolio Builder
# ============================================================
elif page == "💼 Portfolio Builder":
    st.markdown('<div class="main-header">💼 Portfolio Construction</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first using the sidebar.")
        st.stop()

    loader = st.session_state.data_loader

    st.markdown("""
    This module builds optimized investment portfolios using **Modern Portfolio Theory**.
    Portfolios are constructed by optimizing risk-adjusted returns based on historical data.

    Three investor profiles are available, each with different risk-return characteristics.
    """)

    if st.button("🚀 Generate All Portfolios", use_container_width=True, type="primary"):
        with st.spinner("Building optimized portfolios... This may take a moment."):
            portfolio_builder = PortfolioConstructor(
                loader.stock_data,
                loader.sector_mapping,
            )

            portfolios = portfolio_builder.build_all_portfolios()
            st.session_state.portfolio_data = portfolios
            st.session_state.portfolio_builder = portfolio_builder

    if st.session_state.portfolio_data:
        portfolios = st.session_state.portfolio_data
        portfolio_builder = st.session_state.portfolio_builder

        # Three tabs for three profiles
        tab1, tab2, tab3 = st.tabs(
            ["🛡️ Conservative", "⚖️ Balanced", "🚀 Aggressive"]
        )

        for tab, profile_name in [(tab1, "Conservative"), (tab2, "Balanced"), (tab3, "Aggressive")]:
            with tab:
                portfolio = portfolios.get(profile_name, {})
                if not portfolio:
                    st.info("Portfolio data not available.")
                    continue

                st.markdown(f"### {portfolio.get('profile', profile_name)} Portfolio")
                st.markdown(portfolio.get('description', ''))

                st.markdown("---")

                # Metrics
                metrics = portfolio.get('metrics', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    display_metric_card(metrics.get('Expected Annual Return', 'N/A'), "Expected Annual Return",
                                        "success")
                with col2:
                    display_metric_card(metrics.get('Expected Volatility', 'N/A'), "Expected Volatility",
                                        "warning")
                with col3:
                    display_metric_card(metrics.get('Sharpe Ratio', 'N/A'), "Sharpe Ratio", "primary")
                with col4:
                    display_metric_card(metrics.get('Maximum Drawdown', 'N/A'), "Max Drawdown", "info")

                # Additional info
                st.markdown(f"**Investment Horizon:** {portfolio.get('investment_horizon', 'N/A')} | "
                            f"**Rebalancing:** {portfolio.get('rebalancing_frequency', 'N/A')} | "
                            f"**Risk Level:** {portfolio.get('risk_level', 'N/A')}")

                st.markdown("---")

                # Allocation
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### 📊 Portfolio Allocation")
                    allocation = portfolio.get('allocation', [])
                    if allocation:
                        alloc_df = pd.DataFrame(allocation)
                        alloc_df.columns = ['Stock', 'Weight (%)', 'Sector']
                        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

                        # Pie chart of allocation
                        fig = px.pie(
                            alloc_df, values='Weight (%)', names='Stock',
                            title='Portfolio Weights',
                            hole=0.4,
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### 🏭 Sector Diversification")
                    sector_alloc = portfolio.get('sector_allocation', {})
                    if sector_alloc:
                        sector_df = pd.DataFrame(
                            list(sector_alloc.items()),
                            columns=['Sector', 'Allocation (%)']
                        ).sort_values('Allocation (%)', ascending=False)

                        fig_sector = px.bar(
                            sector_df, x='Allocation (%)', y='Sector',
                            orientation='h',
                            title='Sector-wise Allocation',
                            color='Allocation (%)',
                        )
                        fig_sector.update_layout(height=400)
                        st.plotly_chart(fig_sector, use_container_width=True)

                    st.markdown("#### 📝 Stock Selection Rationale")
                    explainer = st.session_state.explainer
                    for item in allocation[:5]:
                        reason = portfolio_builder.get_stock_recommendation_reasoning(
                            item['symbol'], item['weight'] / 100
                        )
                        st.markdown(f"- {reason}")

                # Strategy explanation
                st.markdown("---")
                explanation = st.session_state.explainer.explain_portfolio(
                    portfolio, loader.stock_data, loader.sector_mapping
                )
                st.markdown(f"**Strategy:** {explanation.get('strategy', '')}")
                st.markdown(explanation.get('risk_explanation', ''))

    else:
        st.info("👆 Click 'Generate All Portfolios' to build investment portfolios for different investor profiles.")

    # Efficient Frontier visualization
    if st.session_state.portfolio_data and hasattr(st.session_state, 'portfolio_builder'):
        st.markdown("---")
        st.markdown("### 📈 Efficient Frontier")
        st.markdown("""
        The Efficient Frontier shows optimal portfolios that offer the highest expected return
        for a given level of risk. Portfolios below the frontier are sub-optimal.
        """)

        try:
            with st.spinner("Computing Efficient Frontier..."):
                pb = st.session_state.portfolio_builder
                rets, vols, _ = pb.compute_efficient_frontier(n_points=50)

                if rets:
                    fig_ef = go.Figure()

                    # Efficient frontier line
                    fig_ef.add_trace(go.Scatter(
                        x=[v * 100 for v in vols],
                        y=[r * 100 for r in rets],
                        mode='lines+markers',
                        name='Efficient Frontier',
                        line=dict(color='blue', width=2),
                    ))

                    # Individual stocks
                    for sym in pb.symbols[:20]:
                        vol = np.sqrt(pb.cov_matrix.loc[sym, sym]) * 100
                        ret = pb.mean_returns[sym] * 100
                        fig_ef.add_trace(go.Scatter(
                            x=[vol], y=[ret],
                            mode='markers+text',
                            name=sym,
                            text=[sym],
                            textposition='top center',
                            marker=dict(size=8),
                        ))

                    fig_ef.update_layout(
                        title='Efficient Frontier with Individual Stocks',
                        xaxis_title='Risk (Annualized Volatility %)',
                        yaxis_title='Expected Return (Annual %)',
                        template='plotly_white',
                        height=500,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ef, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute efficient frontier: {e}")


# ============================================================
# PAGE: Risk Assessment
# ============================================================
elif page == "⚠️ Risk Assessment":
    st.markdown('<div class="main-header">⚠️ Risk Assessment Module</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first using the sidebar.")
        st.stop()

    loader = st.session_state.data_loader
    risk_assessor = st.session_state.risk_assessor

    st.markdown("""
    This module provides comprehensive risk analysis for individual stocks,
    helping investors understand the potential downside and risk characteristics
    of their investments.
    """)

    # Stock selector for risk
    selected = st.selectbox(
        "Select Stock for Risk Analysis",
        loader.all_symbols,
        key="risk_stock_selector",
    )

    if st.button("📊 Analyze Risk", use_container_width=True):
        with st.spinner("Computing risk metrics..."):
            df = loader.get_stock(selected)
            if df is None:
                st.error(f"No data for {selected}")
                st.stop()

            risk_report = risk_assessor.assess_stock(df, selected)

            if risk_report:
                # Risk category
                cat = risk_report['risk_category']
                st.markdown(f"## Risk Level: **{cat}**")

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    display_metric_card(f"{risk_report['volatility_annualized']}%", "Annual Volatility", "warning")
                with col2:
                    display_metric_card(str(risk_report['sharpe_ratio']), "Sharpe Ratio",
                                        "success" if risk_report['sharpe_ratio'] > 0.5 else "warning")
                with col3:
                    display_metric_card(str(risk_report['sortino_ratio']), "Sortino Ratio", "info")
                with col4:
                    display_metric_card(f"{risk_report['max_drawdown_pct']}%", "Max Drawdown",
                                        "warning" if risk_report['max_drawdown_pct'] > 30 else "success")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    display_metric_card(f"{risk_report['var_95_daily']}%", "VaR (95% daily)", "info")
                with col2:
                    display_metric_card(f"{risk_report['cvar_95_daily']}%", "CVaR (95% daily)", "warning")
                with col3:
                    display_metric_card(str(risk_report['calmar_ratio']), "Calmar Ratio", "info")
                with col4:
                    display_metric_card(str(risk_report['omega_ratio']), "Omega Ratio",
                                        "success" if risk_report['omega_ratio'] > 1 else "warning")

                # Rolling volatility chart
                st.markdown("---")
                st.markdown("### Rolling Volatility (60-day)")

                rolling_vol = risk_assessor.rolling_volatility(df, 60)
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=df['date'][60:], y=rolling_vol.dropna() * 100,
                    mode='lines', name='60-Day Rolling Volatility',
                    line=dict(color='red', width=1.5),
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
                ))
                fig_vol.add_hline(y=20, line_dash="dash", line_color="orange",
                                  annotation_text="Moderate threshold")
                fig_vol.update_layout(
                    title=f'{selected} - Rolling Annualized Volatility',
                    yaxis_title='Volatility (%)',
                    template='plotly_white', height=400,
                )
                st.plotly_chart(fig_vol, use_container_width=True)

                # Drawdown chart
                st.markdown("### Historical Drawdowns")
                cumulative = (1 + df['close'].pct_change().dropna()).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max * 100

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=df['date'][1:], y=drawdown.values,
                    mode='lines', name='Drawdown',
                    line=dict(color='red', width=1),
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.15)',
                ))
                fig_dd.update_layout(
                    title=f'{selected} - Drawdown Chart',
                    yaxis_title='Drawdown (%)',
                    template='plotly_white', height=350,
                )
                st.plotly_chart(fig_dd, use_container_width=True)

                # Risk explanation
                st.markdown("---")
                st.markdown("### Risk Interpretation")
                explainer = st.session_state.explainer
                explanation = explainer.explain_risk_report(risk_report)
                st.markdown(explanation)

    # Multi-stock comparison
    st.markdown("---")
    st.markdown("### 📊 Cross-Stock Risk Comparison")

    if st.button("Compare Top Stocks by Sharpe Ratio", use_container_width=True):
        with st.spinner("Computing risk metrics for all stocks..."):
            comparison = risk_assessor.compare_stocks(loader.stock_data, top_n=15)
            if not comparison.empty:
                st.dataframe(comparison[[
                    'symbol', 'risk_category', 'volatility_annualized',
                    'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
                    'total_return_pct'
                ]], use_container_width=True, hide_index=True)

                # Risk-return scatter
                fig_comp = px.scatter(
                    comparison,
                    x='volatility_annualized', y='total_return_pct',
                    text='symbol', color='risk_category',
                    size='sharpe_ratio',
                    title='Risk-Return Profile of Stocks',
                    labels={
                        'volatility_annualized': 'Annualized Volatility (%)',
                        'total_return_pct': 'Total Return (%)',
                    },
                )
                fig_comp.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig_comp, use_container_width=True)


# ============================================================
# PAGE: Anomaly Detection
# ============================================================
elif page == "🔍 Anomaly Detection":
    st.markdown('<div class="main-header">🔍 Market Anomaly Detection</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first using the sidebar.")
        st.stop()

    loader = st.session_state.data_loader
    anomaly_detector = st.session_state.anomaly_detector

    st.markdown("""
    This module identifies unusual patterns and events in historical market data,
    including sudden price movements, volume surges, and volatility spikes.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_stock = st.selectbox(
            "Select Stock",
            loader.all_symbols,
            key="anomaly_stock",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("🔍 Scan for Anomalies", use_container_width=True, type="primary")

    if scan_btn:
        with st.spinner(f"Scanning {selected_stock} for anomalies..."):
            df = loader.get_stock(selected_stock)
            report = anomaly_detector.full_anomaly_scan(df, selected_stock)

            if report and report.get('status') != 'Insufficient data':
                st.markdown(f"### Anomaly Report for {selected_stock}")
                st.markdown(f"**Severity Level:** {report.get('anomaly_severity', 'Unknown')}")
                st.markdown(f"**Analysis Period:** {report.get('date_range', 'N/A')}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    price = report.get('price_anomalies', {})
                    display_metric_card(str(price.get('count', 0)), "Price Anomalies",
                                        "warning" if price.get('count', 0) > 20 else "info")
                with col2:
                    vol = report.get('volume_anomalies', {})
                    display_metric_card(str(vol.get('count', 0)), "Volume Anomalies",
                                        "warning" if vol.get('count', 0) > 20 else "info")
                with col3:
                    vspike = report.get('volatility_spikes', {})
                    display_metric_card(str(vspike.get('count', 0)), "Volatility Spikes",
                                        "warning" if vspike.get('count', 0) > 20 else "info")

                # Price anomaly visualization
                st.markdown("---")
                st.markdown("### Price Anomalies Visualization")

                price_anomalies = anomaly_detector.detect_price_anomalies(df, selected_stock)

                if len(price_anomalies) > 0:
                    fig = go.Figure()
                    # Normal price line
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df['close'],
                        mode='lines', name='Close Price',
                        line=dict(color='gray', width=1),
                    ))

                    # Anomaly points
                    extreme_up = price_anomalies[price_anomalies['anomaly_type'] == 'extreme_up']
                    extreme_down = price_anomalies[price_anomalies['anomaly_type'] == 'extreme_down']

                    if len(extreme_up) > 0:
                        fig.add_trace(go.Scatter(
                            x=extreme_up['date'], y=extreme_up['close'],
                            mode='markers', name='Extreme Up',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                        ))
                    if len(extreme_down) > 0:
                        fig.add_trace(go.Scatter(
                            x=extreme_down['date'], y=extreme_down['close'],
                            mode='markers', name='Extreme Down',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                        ))

                    fig.update_layout(
                        title=f'{selected_stock} - Price Anomalies',
                        template='plotly_white', height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Volume anomalies
                st.markdown("### Volume Anomalies")
                volume_anomalies = anomaly_detector.detect_volume_anomalies(df, selected_stock)

                if len(volume_anomalies) > 0:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=df['date'], y=df['volume'],
                        name='Volume', marker_color='lightblue',
                    ))
                    fig_vol.add_trace(go.Scatter(
                        x=volume_anomalies['date'], y=volume_anomalies['volume'],
                        mode='markers', name='Volume Surge',
                        marker=dict(color='red', size=8, symbol='x'),
                    ))
                    fig_vol.update_layout(
                        title=f'{selected_stock} - Volume Anomalies',
                        template='plotly_white', height=350,
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                # Anomaly explanation
                st.markdown('<div class="warning-box">'
                            '<strong>⚠️ Understanding Anomalies:</strong> Anomalies represent statistically '
                            'unusual market behavior. While some anomalies may correspond to known events '
                            '(earnings announcements, policy changes, global events), others may indicate '
                            'market inefficiencies or data quality issues. Investors should investigate '
                            'anomalies further before making decisions based on them.'
                            '</div>', unsafe_allow_html=True)

    # Market-wide events
    st.markdown("---")
    st.markdown("### 🌐 Market-Wide Event Detection")
    st.markdown("Identify dates where multiple stocks showed anomalies simultaneously (indicating market-wide events).")

    if st.button("Detect Market-Wide Events", use_container_width=True):
        with st.spinner("Scanning all stocks for market-wide events..."):
            market_events = anomaly_detector.detect_market_wide_events(loader.stock_data)

            if not market_events.empty:
                st.markdown(f"Found {len(market_events)} dates with multi-stock anomalies")

                # Show top events
                top_events = market_events.head(20)
                st.dataframe(
                    top_events[['date', 'stock_count', 'pct_stocks_affected']],
                    use_container_width=True, hide_index=True,
                )

                # Bar chart of events
                fig_events = px.bar(
                    top_events.sort_values('date'),
                    x='date', y='stock_count',
                    title='Market-Wide Anomaly Events',
                    labels={'stock_count': 'Number of Stocks Affected'},
                    color='stock_count',
                )
                fig_events.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig_events, use_container_width=True)

                st.markdown('<div class="info-box">'
                            'These dates likely correspond to major market events such as the 2008 financial '
                            'crisis, COVID-19 market crash (March 2020), Union Budget days, election results, '
                            'and RBI policy announcements.'
                            '</div>', unsafe_allow_html=True)
            else:
                st.info("No significant market-wide events detected.")


# ============================================================
# PAGE: Forecasting
# ============================================================
elif page == "🔮 Forecasting":
    st.markdown('<div class="main-header">🔮 Advanced Forecasting</div>',
                unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first using the sidebar.")
        st.stop()

    loader = st.session_state.data_loader
    forecaster = st.session_state.forecaster

    st.markdown("""
    This module provides multi-method forecasting for stock prices and volatility.
    It combines statistical models, Monte Carlo simulation, and trend analysis
    to generate comprehensive forecasts with confidence intervals.
    """)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected = st.selectbox("Select Stock", loader.all_symbols, key="forecast_stock")
    with col2:
        horizon = st.selectbox("Forecast Horizon (days)", [7, 15, 30, 60, 90], index=2)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_forecast = st.button("🔮 Generate Forecast", use_container_width=True, type="primary")

    if run_forecast:
        with st.spinner("Running multi-method forecasting..."):
            df = loader.get_stock(selected)
            forecast_result = forecaster.full_forecast(df, horizon)

            if forecast_result and 'error' not in forecast_result:
                st.markdown(f"### Forecast Results for {selected} ({horizon}-day horizon)")

                # Ensemble forecast summary
                ensemble = forecast_result.get('ensemble_forecast', {})
                if ensemble:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        emoji = "🟢" if ensemble['direction'] == 'UP' else "🔴"
                        display_metric_card(f"{emoji} {ensemble['direction']}", "Direction",
                                            "success" if ensemble['direction'] == 'UP' else "warning")
                    with col2:
                        display_metric_card(f"{ensemble['expected_return_pct']:+.2f}%", "Expected Return", "primary")
                    with col3:
                        display_metric_card(f"₹{ensemble['expected_price']:.2f}", "Expected Price", "info")

                st.markdown("---")

                # Monte Carlo simulation
                mc = forecast_result.get('monte_carlo', {})
                if mc:
                    st.markdown("### 🎲 Monte Carlo Simulation")
                    st.markdown(f"Based on **{mc.get('method', '')}** with annualized drift of "
                                f"**{mc.get('annualized_drift', 0)}%** and volatility of "
                                f"**{mc.get('annualized_volatility', 0)}%**")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        display_metric_card(f"₹{mc.get('expected_final_price', 0):.2f}", "Expected Price", "primary")
                    with col2:
                        display_metric_card(f"{mc.get('expected_return_pct', 0):+.2f}%", "Expected Return",
                                            "success" if mc.get('expected_return_pct', 0) > 0 else "warning")
                    with col3:
                        display_metric_card(f"{mc.get('prob_positive_return', 0)}%", "Prob. of Gain", "info")
                    with col4:
                        display_metric_card(f"₹{mc.get('median_final_price', 0):.2f}", "Median Price", "info")

                    # Percentile table
                    st.markdown("#### Return Distribution Percentiles")
                    ret_perc = mc.get('return_percentiles', {})
                    if ret_perc:
                        perc_df = pd.DataFrame([
                            {'Percentile': k, 'Return (%)': v} for k, v in ret_perc.items()
                        ])
                        st.dataframe(perc_df, use_container_width=True, hide_index=True)

                    # Monte Carlo paths visualization
                    st.markdown("#### Simulated Price Paths")
                    mean_path = mc.get('mean_path', [])
                    upper = mc.get('upper_band_95', [])
                    lower = mc.get('lower_band_5', [])

                    if mean_path:
                        fig_mc = go.Figure()
                        future_dates = pd.date_range(
                            start=df['date'].iloc[-1] + timedelta(days=1),
                            periods=horizon, freq='B'
                        )

                        # Confidence interval
                        fig_mc.add_trace(go.Scatter(
                            x=list(future_dates) + list(future_dates[::-1]),
                            y=list(upper) + list(lower[::-1]),
                            fill='toself', fillcolor='rgba(0,100,200,0.2)',
                            line=dict(color='rgba(0,0,0,0)'),
                            name='90% Confidence Interval',
                        ))

                        # Mean path
                        fig_mc.add_trace(go.Scatter(
                            x=future_dates, y=mean_path,
                            mode='lines', name='Expected Path',
                            line=dict(color='blue', width=2),
                        ))

                        # Historical line
                        hist_dates = df['date'].iloc[-90:]
                        hist_prices = df['close'].iloc[-90:]
                        fig_mc.add_trace(go.Scatter(
                            x=hist_dates, y=hist_prices,
                            mode='lines', name='Historical',
                            line=dict(color='gray', width=1.5),
                        ))

                        fig_mc.update_layout(
                            title=f'{selected} - Monte Carlo Price Simulation',
                            xaxis_title='Date',
                            yaxis_title='Price (Rs.)',
                            template='plotly_white', height=450,
                        )
                        st.plotly_chart(fig_mc, use_container_width=True)

                st.markdown("---")

                # Trend forecast
                trend = forecast_result.get('trend_forecast', {})
                if trend:
                    st.markdown("### 📈 Trend Regression Forecast")
                    st.markdown(f"**Trend Direction:** {trend.get('trend_direction', 'N/A').title()} | "
                                f"**R²:** {trend.get('r_squared', 'N/A')}")

                    trend_vals = trend.get('forecast', [])
                    if trend_vals:
                        future_dates = pd.date_range(
                            start=df['date'].iloc[-1] + timedelta(days=1),
                            periods=len(trend_vals), freq='B'
                        )

                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=future_dates, y=trend.get('upper_band', []),
                            mode='lines', name='Upper 95%',
                            line=dict(color='lightgreen', dash='dash'),
                        ))
                        fig_trend.add_trace(go.Scatter(
                            x=future_dates, y=trend_vals,
                            mode='lines', name='Forecast',
                            line=dict(color='green', width=2),
                        ))
                        fig_trend.add_trace(go.Scatter(
                            x=future_dates, y=trend.get('lower_band', []),
                            mode='lines', name='Lower 95%',
                            line=dict(color='lightcoral', dash='dash'),
                            fill='tonexty', fillcolor='rgba(0,255,0,0.1)',
                        ))
                        fig_trend.update_layout(
                            title='Linear Trend Forecast with Confidence Bands',
                            template='plotly_white', height=400,
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)

                # Volatility forecast
                vol_fc = forecast_result.get('volatility_forecast', {})
                if vol_fc:
                    st.markdown("---")
                    st.markdown("### 📉 Volatility Forecast")
                    st.markdown(f"**Current Annualized Volatility:** {vol_fc.get('current_annualized_volatility', 'N/A')}%")
                    st.markdown(f"**Long-term Average Volatility:** {vol_fc.get('long_term_volatility', 'N/A')}%")

                    vol_path = vol_fc.get('forecast_volatility_path', [])
                    if vol_path:
                        future_dates = pd.date_range(
                            start=df['date'].iloc[-1] + timedelta(days=1),
                            periods=len(vol_path), freq='B'
                        )
                        fig_vol_fc = go.Figure()
                        fig_vol_fc.add_trace(go.Scatter(
                            x=future_dates, y=vol_path,
                            mode='lines+markers', name='Forecast Volatility',
                            line=dict(color='orange', width=2),
                            fill='tozeroy', fillcolor='rgba(255,165,0,0.1)',
                        ))
                        fig_vol_fc.add_hline(
                            y=float(vol_fc.get('long_term_volatility', 20)),
                            line_dash="dash", line_color="gray",
                            annotation_text="Long-term avg"
                        )
                        fig_vol_fc.update_layout(
                            title='Volatility Forecast Path (Annualized %)',
                            yaxis_title='Volatility (%)',
                            template='plotly_white', height=350,
                        )
                        st.plotly_chart(fig_vol_fc, use_container_width=True)

                # Disclaimer
                st.markdown('<div class="warning-box">'
                            '<strong>⚠️ Forecast Disclaimer:</strong> All forecasts are based on historical '
                            'data and statistical models. They are not guarantees of future performance. '
                            'Financial markets are influenced by many factors including economic conditions, '
                            'policy changes, and global events that models cannot predict. Use forecasts '
                            'as one of many inputs to your investment decision process, not as the sole basis.'
                            '</div>', unsafe_allow_html=True)

            else:
                st.error("Could not generate forecasts. The stock may have insufficient data.")


# ============================================================
# PAGE: About & Guide
# ============================================================
elif page == "📋 About & Guide":
    st.markdown('<div class="main-header">📋 About & User Guide</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## About This Project

        This platform was developed as part of **Open Projects 2026** organized by the
        **Cultural Council, IIT Roorkee**.

        ### Developer
        - **Name:** Praneshwar Kannan Kommiya
        - **ID:** 23117102
        - **Program:** B.Tech Mechanical Engineering (4th Year)
        - **Institution:** Indian Institute of Technology, Roorkee

        ### Project: Data-Driven Investment Intelligence Using NIFTY-50 Market Data

        #### Objective
        To develop an AI-powered investment intelligence platform capable of transforming
        historical market data into meaningful insights that assist investors in making
        informed decisions.

        #### Key Features
        1. **Stock Predictor Engine** — ML-based forecasting using XGBoost and LSTM
        2. **Portfolio Construction** — Optimized portfolios for different risk profiles
        3. **Risk Assessment** — Comprehensive risk metrics and analysis
        4. **Anomaly Detection** — Identification of unusual market patterns
        5. **Advanced Forecasting** — Multi-method forecasting with Monte Carlo simulation
        6. **Explainable AI** — Transparent reasoning behind predictions and recommendations

        #### Dataset
        NIFTY-50 Stock Market Dataset (Jan 2000 – Apr 2021) from Kaggle.
        """)

    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Python** — Core programming language
        - **Streamlit** — Web application framework
        - **Plotly** — Interactive visualizations
        - **XGBoost** — Gradient boosting for predictions
        - **TensorFlow/Keras** — Deep learning (LSTM)
        - **Scikit-learn** — Machine learning utilities
        - **Pandas/NumPy** — Data manipulation
        - **SciPy** — Statistical optimization
        """)

    st.markdown("---")
    st.markdown("### 📖 User Guide")

    with st.expander("1️⃣ Getting Started", expanded=True):
        st.markdown("""
        1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
        2. Extract all CSV files into the `data/` folder
        3. Install dependencies: `pip install -r requirements.txt`
        4. Launch the app: `streamlit run app.py`
        5. Use the sidebar to load your data and navigate between modules
        """)

    with st.expander("2️⃣ Stock Analysis Module"):
        st.markdown("""
        - Select any NIFTY-50 stock from the dropdown
        - Choose a prediction horizon (7-90 days)
        - View **candlestick charts** with Bollinger Bands and moving averages
        - Check **technical indicators** (RSI, MACD, ATR)
        - Get **ML predictions** with feature importance explanations
        - Review **risk metrics** including VaR, Sharpe Ratio, and drawdowns
        """)

    with st.expander("3️⃣ Portfolio Builder"):
        st.markdown("""
        - Click "Generate All Portfolios" to build three optimized portfolios
        - **Conservative:** Low risk, stable returns, suitable for capital preservation
        - **Balanced:** Moderate risk, diversified across sectors
        - **Aggressive:** Higher risk, growth-oriented, for long-term investors
        - Each portfolio shows allocation, sector breakdown, and selection rationale
        - The **Efficient Frontier** visualization shows optimal risk-return tradeoffs
        """)

    with st.expander("4️⃣ Risk Assessment"):
        st.markdown("""
        - Analyze individual stock risk with 10+ metrics
        - Key metrics explained:
          - **Volatility** — How much the price fluctuates
          - **Sharpe Ratio** — Return per unit of risk (>1 is good)
          - **Sortino Ratio** — Like Sharpe but only penalizes downside
          - **VaR (95%)** — Maximum daily loss 95% of the time
          - **Max Drawdown** — Worst peak-to-trough decline
          - **Omega Ratio** — Ratio of gains to losses (>1 is favorable)
        - Compare risk across multiple stocks
        """)

    with st.expander("5️⃣ Anomaly Detection"):
        st.markdown("""
        - Detects three types of anomalies:
          - **Price anomalies** — Unusually large daily moves
          - **Volume anomalies** — Abnormal trading activity
          - **Volatility spikes** — Sudden increases in price fluctuation
        - Market-wide event detection finds dates where many stocks moved unusually
        """)

    with st.expander("6️⃣ Forecasting"):
        st.markdown("""
        - **Trend Regression** — Linear trend projection with confidence bands
        - **Exponential Smoothing** — Holt's method with trend component
        - **Monte Carlo Simulation** — 500+ simulated price paths with probability distributions
        - **Volatility Forecast** — GARCH-style volatility path estimation
        - Ensemble forecast combines multiple methods
        """)

    st.markdown("---")
    st.markdown("""
    ### ⚠️ Disclaimer

    This platform is for **educational and informational purposes only**. It does not
    constitute financial advice. All predictions and recommendations are based on
    historical data and statistical models, which cannot guarantee future performance.

    Investment decisions should be made considering your personal financial situation,
    risk tolerance, and investment goals. Past performance does not guarantee future results.

    ---
    *Built with ❤️ at IIT Roorkee | Open Projects 2026 | Cultural Council*
    """)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown('<div class="footer">'
            '📊 <strong>NIFTY-50 Investment Intelligence Platform</strong><br>'
            'Built by <strong>Praneshwar Kannan Kommiya</strong> (23117102) | '
            'B.Tech Mechanical Engineering - 4Y<br>'
            'Open Projects 2026 · Cultural Council · IIT Roorkee<br>'
            '<small>For educational purposes only. Not financial advice.</small>'
            '</div>', unsafe_allow_html=True)
