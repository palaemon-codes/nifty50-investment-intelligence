"""
Data Loader Module
Handles downloading, loading, cleaning, and preprocessing of NIFTY-50 stock data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import DATA_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class NIFTY50DataLoader:
    """
    Loads and manages NIFTY-50 stock market data.
    Works with CSV files downloaded from Kaggle (NIFTY-50 dataset).
    
    Expected CSV format (from the Kaggle dataset):
    - Each CSV named after the stock symbol
    - Columns: Date, Open, High, Low, Close, Volume, Turnover (or similar names)
    """

    def __init__(self, data_path=None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the folder containing stock CSV files.
            Defaults to DATA_DIR from config.
        """
        self.data_path = data_path or DATA_DIR
        self.stock_data = {}           # Dictionary: symbol -> DataFrame
        self.company_metadata = None   # DataFrame with company info
        self.all_symbols = []
        self.sector_mapping = {}

    def load_from_folder(self, folder_path=None):
        """
        Load all stock CSV files from the nifty_data folder.
        Skips NIFTY50_all.csv (combined file) and stock_metadata.csv (company info).
        """
        folder = folder_path or self.data_path

        if not os.path.exists(folder):
            print(f"  Data folder not found: {folder}")
            print(f"  Expected location: ~/Downloads/nifty_data/")
            print("  Make sure the nifty_data folder is placed in your Downloads directory.")
            return False

        # Get all CSV files, skip the combined file and metadata
        skip_files = {'NIFTY50_all.csv', 'stock_metadata.csv'}
        csv_files = [f for f in os.listdir(folder)
                     if f.endswith('.csv') and f not in skip_files]

        if not csv_files:
            print(f"  No stock CSV files found in {folder}")
            return False

        print(f"  Found {len(csv_files)} stock CSV files")

        loaded_count = 0
        for filename in csv_files:
            symbol = filename.replace('.csv', '').strip().upper()
            filepath = os.path.join(folder, filename)

            try:
                df = pd.read_csv(filepath)
                df = self._clean_stock_data(df, symbol)
                if df is not None and len(df) > 0:
                    self.stock_data[symbol] = df
                    loaded_count += 1
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")

        self.all_symbols = sorted(list(self.stock_data.keys()))
        print(f"  Successfully loaded {loaded_count} stocks")
        return loaded_count > 0

    def _clean_stock_data(self, df, symbol):
        """
        Clean and standardize a single stock's DataFrame.
        The dataset has Title Case columns: Date, Symbol, Open, High, Low, Close, Volume, etc.
        """
        if df.empty:
            return None

        # Strip whitespace from column names but keep case for mapping
        df.columns = [col.strip() for col in df.columns]

        # Map the actual Kaggle dataset column names (Title Case) to lowercase standard names
        column_mapping = {
            'Date': 'date',
            'Symbol': 'symbol',
            'Series': 'series',
            'Prev Close': 'prev_close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Last': 'last',
            'Close': 'close',
            'VWAP': 'vwap',
            'Volume': 'volume',
            'Turnover': 'turnover',
            'Trades': 'trades',
            'Deliverable Volume': 'deliverable_volume',
            '%Deliverble': 'pct_deliverable',
        }

        # Apply renaming — keep only columns that exist
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        df.rename(columns=rename_dict, inplace=True)

        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            print(f"  Warning: {symbol} missing columns: {missing}")
            if 'date' in missing:
                return None
            for col in missing:
                df[col] = np.nan

        # Convert date column — this dataset uses YYYY-MM-DD format
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            print(f"  Warning: Could not parse dates for {symbol}")
            return None

        # Drop rows with invalid dates
        df = df.dropna(subset=['date']).copy()

        # Ensure numeric types for price/volume columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where close price is missing
        df = df.dropna(subset=['close']).copy()

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Add symbol column if not already present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol

        # Remove duplicate dates (keep first)
        df = df.drop_duplicates(subset=['date'], keep='first')

        # Basic sanity checks
        if len(df) < 30:
            print(f"  Warning: {symbol} has only {len(df)} rows, skipping")
            return None

        # Ensure high >= low, high >= open, high >= close, low <= open, low <= close
        if 'high' in df.columns and 'low' in df.columns:
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            df['low'] = df[['low', 'open', 'close']].min(axis=1)

        return df

    def load_company_metadata(self, metadata_path=None):
        """
        Load company metadata from stock_metadata.csv in the nifty_data folder.
        The CSV has columns: Company Name, Industry, Symbol, Series, ISIN Code.
        """
        # Try to find stock_metadata.csv
        if metadata_path is None:
            metadata_path = os.path.join(self.data_path, 'stock_metadata.csv')

        if os.path.exists(metadata_path):
            try:
                meta_df = pd.read_csv(metadata_path)
                # Columns: Company Name, Industry, Symbol, Series, ISIN Code
                if 'Symbol' in meta_df.columns and 'Industry' in meta_df.columns:
                    self.company_metadata = meta_df
                    self.sector_mapping = dict(zip(
                        meta_df['Symbol'].str.upper(),
                        meta_df['Industry']
                    ))
                    print(f"  Loaded metadata for {len(self.sector_mapping)} companies from stock_metadata.csv")
                    return
            except Exception as e:
                print(f"  Could not parse stock_metadata.csv: {e}")

        # Fallback: built-in sector mapping for major NIFTY-50 constituents
        self.sector_mapping = {
            'RELIANCE': 'ENERGY',
            'TCS': 'IT', 'HDFCBANK': 'FINANCIAL SERVICES',
            'INFY': 'IT', 'ICICIBANK': 'FINANCIAL SERVICES',
            'HINDUNILVR': 'CONSUMER GOODS', 'KOTAKBANK': 'FINANCIAL SERVICES',
            'ITC': 'CONSUMER GOODS', 'SBIN': 'FINANCIAL SERVICES',
            'BHARTIARTL': 'TELECOM', 'BAJFINANCE': 'FINANCIAL SERVICES',
            'WIPRO': 'IT', 'ASIANPAINT': 'CONSUMER GOODS',
            'MARUTI': 'AUTOMOBILE', 'SUNPHARMA': 'PHARMA',
            'TITAN': 'CONSUMER GOODS', 'AXISBANK': 'FINANCIAL SERVICES',
            'LT': 'CONSTRUCTION', 'ONGC': 'ENERGY', 'NTPC': 'ENERGY',
            'POWERGRID': 'ENERGY', 'NESTLEIND': 'CONSUMER GOODS',
            'ULTRACEMCO': 'CEMENT', 'TECHM': 'IT', 'HCLTECH': 'IT',
            'TATAMOTORS': 'AUTOMOBILE', 'M&M': 'AUTOMOBILE',
            'TATASTEEL': 'METALS', 'JSWSTEEL': 'METALS',
            'HINDALCO': 'METALS', 'GRASIM': 'CEMENT',
            'DRREDDY': 'PHARMA', 'CIPLA': 'PHARMA', 'DIVISLAB': 'PHARMA',
            'BRITANNIA': 'CONSUMER GOODS', 'HDFCLIFE': 'FINANCIAL SERVICES',
            'BAJAJ-AUTO': 'AUTOMOBILE', 'EICHERMOT': 'AUTOMOBILE',
            'HEROMOTOCO': 'AUTOMOBILE', 'COALINDIA': 'ENERGY',
            'BPCL': 'ENERGY', 'IOC': 'ENERGY', 'SHREECEM': 'CEMENT',
            'ADANIPORTS': 'SERVICES', 'UPL': 'CHEMICALS',
            'INDUSINDBK': 'FINANCIAL SERVICES', 'TATACONSUM': 'CONSUMER GOODS',
            'BAJAJFINSV': 'FINANCIAL SERVICES', 'SBILIFE': 'FINANCIAL SERVICES',
            'HDFC': 'FINANCIAL SERVICES', 'MM': 'AUTOMOBILE',
            'GAIL': 'ENERGY', 'VEDL': 'METALS', 'ZEEL': 'MEDIA & ENTERTAINMENT',
            'INFRATEL': 'TELECOM',
        }

        print(f"  Using built-in sector mapping with {len(self.sector_mapping)} companies")

    def get_stock(self, symbol):
        """Return DataFrame for a specific stock symbol."""
        symbol = symbol.upper()
        return self.stock_data.get(symbol)

    def get_sector(self, symbol):
        """Get the sector for a given stock symbol."""
        symbol = symbol.upper()
        return self.sector_mapping.get(symbol, "Unknown")

    def get_stocks_by_sector(self, sector):
        """Return list of stock symbols in a given sector."""
        return [s for s in self.all_symbols
                if self.sector_mapping.get(s, "Unknown") == sector]

    def get_all_sectors(self):
        """Return list of all unique sectors."""
        sectors = set()
        for sym in self.all_symbols:
            sectors.add(self.sector_mapping.get(sym, "Unknown"))
        return sorted(list(sectors))

    def get_combined_data(self, symbols=None):
        """
        Return a combined DataFrame with data for multiple stocks.
        Useful for portfolio-level analysis.
        """
        if symbols is None:
            symbols = self.all_symbols

        combined = []
        for sym in symbols:
            df = self.get_stock(sym)
            if df is not None:
                combined.append(df)

        if not combined:
            return pd.DataFrame()

        result = pd.concat(combined, ignore_index=True)
        return result

    def get_date_range_summary(self):
        """Print summary of available date ranges for each stock."""
        print("\n  Stock Data Availability Summary:")
        print("  " + "-" * 70)
        print(f"  {'Symbol':<15} {'Start Date':<15} {'End Date':<15} {'Rows':<8} {'Sector'}")
        print("  " + "-" * 70)

        for sym in sorted(self.all_symbols):
            df = self.stock_data[sym]
            if df is not None and len(df) > 0:
                start = df['date'].min().strftime('%Y-%m-%d')
                end = df['date'].max().strftime('%Y-%m-%d')
                sector = self.sector_mapping.get(sym, "Unknown")
                print(f"  {sym:<15} {start:<15} {end:<15} {len(df):<8} {sector}")

        print("  " + "-" * 70)

    def get_summary_stats(self):
        """Return summary statistics for all loaded stocks."""
        if not self.stock_data:
            return pd.DataFrame()

        summaries = []
        for sym, df in self.stock_data.items():
            if df is not None and len(df) > 0:
                latest_close = df['close'].iloc[-1] if 'close' in df.columns else np.nan
                avg_volume = df['volume'].mean() if 'volume' in df.columns else np.nan

                # Calculate returns
                returns = df['close'].pct_change().dropna()
                annual_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else np.nan
                total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) if 'close' in df.columns else np.nan

                summaries.append({
                    'Symbol': sym,
                    'Sector': self.sector_mapping.get(sym, 'Unknown'),
                    'Latest Close': round(latest_close, 2) if not np.isnan(latest_close) else np.nan,
                    'Avg Daily Volume': int(avg_volume) if not np.isnan(avg_volume) else np.nan,
                    'Annualized Volatility': round(annual_vol * 100, 2) if not np.isnan(annual_vol) else np.nan,
                    'Total Return': round(total_return * 100, 2) if not np.isnan(total_return) else np.nan,
                    'Data Points': len(df),
                    'Start Date': df['date'].min(),
                    'End Date': df['date'].max(),
                })

        return pd.DataFrame(summaries)


# ============================================================
# Utility function for quick loading
# ============================================================

def load_nifty50_data(data_folder=None):
    """
    Convenience function to load all NIFTY-50 data.
    
    Usage:
        loader = load_nifty50_data("path/to/data/folder")
        # Then access: loader.stock_data, loader.all_symbols, etc.
    """
    loader = NIFTY50DataLoader(data_folder)
    loader.load_company_metadata()

    if data_folder:
        loader.load_from_folder(data_folder)
    else:
        loader.load_from_folder()

    if loader.all_symbols:
        loader.get_date_range_summary()

    return loader
