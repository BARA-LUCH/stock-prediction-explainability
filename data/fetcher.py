"""
data/fetcher.py
Fetches historical stock, ETF, and crypto data from Yahoo Finance.
Covers US stocks, Israeli stocks, and major cryptocurrencies.
5 years of daily OHLCV data per asset.
Robust error handling + data validation throughout.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# ── Asset Universe ────────────────────────────────────────────────────────────

US_STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "XOM": "ExxonMobil",
    "BAC": "Bank of America",
    "PG": "Procter & Gamble",
    "MA": "Mastercard",
    "HD": "Home Depot",
    "CVX": "Chevron",
    "ABBV": "AbbVie",
    "PFE": "Pfizer",
    "NFLX": "Netflix",
}

ISRAELI_STOCKS = {
    "CHKP": "Check Point Software",
    "NICE": "NICE Systems",
    "CYBR": "CyberArk",
    "WIX": "Wix.com",
    "MNDY": "Monday.com",
    "GLBE": "Global-E Online",
    "CEVA": "CEVA Inc",
    "KRNT": "Kornit Digital",
    "TEVA": "Teva Pharmaceutical",
    "ICL": "ICL Group",
}

CRYPTO = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "SOL-USD": "Solana",
    "ADA-USD": "Cardano",
    "XRP-USD": "XRP",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche",
    "DOT-USD": "Polkadot",
    "MATIC-USD": "Polygon",
}

ETFs = {
    "SPY": "S&P 500 ETF",
    "QQQ": "NASDAQ ETF",
    "IWM": "Russell 2000 ETF",
    "GLD": "Gold ETF",
    "TLT": "Treasury Bond ETF",
}

ALL_ASSETS = {**US_STOCKS, **ISRAELI_STOCKS, **CRYPTO, **ETFs}

MARKET_CATEGORIES = {
    "US Stocks": list(US_STOCKS.keys()),
    "Israeli Stocks": list(ISRAELI_STOCKS.keys()),
    "Crypto": list(CRYPTO.keys()),
    "ETFs": list(ETFs.keys()),
}

MIN_ROWS = 100
REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


# ── Data Validation ───────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, ticker: str = "") -> tuple:
    """
    Validate a fetched DataFrame.
    Returns (is_valid: bool, reason: str)
    """
    if df is None:
        return False, "DataFrame is None"
    if df.empty:
        return False, "DataFrame is empty"
    if len(df) < MIN_ROWS:
        return False, f"Too few rows: {len(df)} (minimum {MIN_ROWS})"
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    if df["Close"].isna().all():
        return False, "All Close prices are NaN"
    if (df["Close"] == 0).mean() > 0.1:
        return False, "Too many zero prices"
    if (df["Close"] < 0).any():
        return False, "Negative prices detected"
    return True, "OK"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize a raw yfinance DataFrame.
    Handles multi-level columns, gaps, duplicates, and outliers.
    """
    if df is None or df.empty:
        return df

    # Flatten multi-level columns (yfinance v0.2+ sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Keep only OHLCV
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # Forward fill small gaps (weekends/holidays — max 3 days)
    df = df.ffill(limit=3)

    # Drop rows where Close is still NaN
    df = df.dropna(subset=["Close"])

    # Remove extreme price outliers (>10x or <0.1x rolling median)
    rolling_med = df["Close"].rolling(20, min_periods=1).median()
    ratio = df["Close"] / rolling_med.replace(0, np.nan)
    mask = (ratio > 0.1) & (ratio < 10)
    df = df[mask]

    # Fill NaN volume with 0
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    return df


# ── Fetch Functions ───────────────────────────────────────────────────────────

def fetch_single_asset(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single asset.
    Retries up to 3 times with exponential backoff.
    Returns cleaned/validated DataFrame or None.
    """
    end = datetime.today()
    start = end - timedelta(days=years * 365)

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                actions=False,
            )

            if df is None or df.empty:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    continue
                return None

            df = clean_dataframe(df)

            is_valid, reason = validate_dataframe(df, ticker)
            if not is_valid:
                if attempt < 2:
                    time.sleep(1)
                    continue
                print(f"  ⚠️  {ticker} validation failed: {reason}")
                return None

            df["Ticker"] = ticker
            df["Name"] = ALL_ASSETS.get(ticker, ticker)
            return df

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  ❌ {ticker} failed after 3 attempts: {type(e).__name__}: {e}")
                return None

    return None


def load_asset(ticker: str, data_path: str = "data/raw") -> pd.DataFrame:
    """
    Load a single asset from cache, or fetch fresh if cache is missing/stale/invalid.
    """
    csv_path = os.path.join(data_path, f"{ticker}.csv")

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = clean_dataframe(df)
            is_valid, reason = validate_dataframe(df, ticker)
            if is_valid:
                days_old = (datetime.today() - df.index[-1]).days
                if days_old < 3:
                    return df
            print(f"  ⚠️  Cache for {ticker} invalid or stale ({reason}), refetching...")
        except Exception as e:
            print(f"  ⚠️  Could not load cache for {ticker}: {e}, refetching...")

    return fetch_single_asset(ticker)


def fetch_all_assets(years: int = 5, save_path: str = "data/raw") -> dict:
    """
    Fetch all assets in the universe. Skips failed assets gracefully.
    Returns {ticker: DataFrame} and saves CSVs locally.
    """
    os.makedirs(save_path, exist_ok=True)
    results = {}
    failed = []

    print(f"📥 Fetching {len(ALL_ASSETS)} assets ({years} years of data)...")

    for i, (ticker, name) in enumerate(ALL_ASSETS.items(), 1):
        csv_path = os.path.join(save_path, f"{ticker}.csv")

        # Use cache if available and fresh
        if os.path.exists(csv_path):
            try:
                cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                cached = clean_dataframe(cached)
                is_valid, _ = validate_dataframe(cached, ticker)
                if is_valid and (datetime.today() - cached.index[-1]).days < 3:
                    results[ticker] = cached
                    print(f"  [{i:02d}/{len(ALL_ASSETS)}] {ticker} — cache ✅")
                    continue
            except Exception:
                pass

        print(f"  [{i:02d}/{len(ALL_ASSETS)}] Fetching {ticker} ({name})...")
        df = fetch_single_asset(ticker, years)

        if df is not None:
            try:
                df.to_csv(csv_path)
            except Exception as e:
                print(f"  ⚠️  Could not save {ticker}: {e}")
            results[ticker] = df
            print(f"  [{i:02d}/{len(ALL_ASSETS)}] {ticker} — {len(df)} rows ✅")
        else:
            failed.append(ticker)
            print(f"  [{i:02d}/{len(ALL_ASSETS)}] {ticker} — FAILED ❌")

    print(f"\n✅ Success: {len(results)}/{len(ALL_ASSETS)}")
    if failed:
        print(f"⚠️  Failed tickers: {', '.join(failed)}")

    return results


def get_market_summary(data: dict) -> pd.DataFrame:
    """Build a summary table of all fetched assets."""
    rows = []
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        try:
            category = next(
                (cat for cat, tickers in MARKET_CATEGORIES.items() if ticker in tickers),
                "Other"
            )
            first_p = df["Close"].iloc[0]
            last_p = df["Close"].iloc[-1]
            ret = (last_p / first_p - 1) * 100 if first_p > 0 else 0
            rows.append({
                "Ticker": ticker,
                "Name": ALL_ASSETS.get(ticker, ticker),
                "Category": category,
                "Start": df.index[0].strftime("%Y-%m-%d"),
                "End": df.index[-1].strftime("%Y-%m-%d"),
                "Days": len(df),
                "Latest Price": round(last_p, 2),
                "5Y Return %": round(ret, 1),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Category").reset_index(drop=True)
