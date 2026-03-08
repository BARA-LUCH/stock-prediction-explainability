"""
features/engineer.py
Generates 30+ technical indicators and features for ML models.
Covers trend, momentum, volatility, volume, and pattern features.
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """SMA and EMA across multiple timeframes."""
    for window in [5, 10, 20, 50, 200]:
        df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    # Golden/Death cross signals
    df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["Price_Above_SMA20"] = (df["Close"] > df["SMA_20"]).astype(int)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)
    df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD, Signal line, and Histogram."""
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Bullish"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Bollinger Bands and derived features."""
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["BB_Upper"] = sma + (2 * std)
    df["BB_Lower"] = sma - (2 * std)
    df["BB_Middle"] = sma
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)
    df["BB_Squeeze"] = (df["BB_Width"] < df["BB_Width"].rolling(50).mean()).astype(int)
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average True Range (volatility measure)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"]
    return df


def add_stochastic(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Stochastic Oscillator %K and %D."""
    low_min = df["Low"].rolling(window).min()
    high_max = df["High"].rolling(window).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based indicators."""
    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA20"] + 1e-10)
    df["Volume_Spike"] = (df["Volume_Ratio"] > 2.0).astype(int)

    # On-Balance Volume
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_SMA"] = pd.Series(obv, index=df.index).rolling(20).mean()
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price-derived features."""
    df["Returns_1d"] = df["Close"].pct_change(1)
    df["Returns_5d"] = df["Close"].pct_change(5)
    df["Returns_20d"] = df["Close"].pct_change(20)

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_20d"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)

    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Price momentum
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar-based features."""
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Quarter"] = df.index.quarter
    df["IsMonday"] = (df.index.dayofweek == 0).astype(int)
    df["IsFriday"] = (df.index.dayofweek == 4).astype(int)
    return df


def create_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Create prediction targets:
    - Target_Direction: 1 if price goes up in `horizon` days, 0 if down
    - Target_Return: actual % return over `horizon` days
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["Target_Return"] = future_return
    df["Target_Direction"] = (future_return > 0).astype(int)
    return df


def engineer_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Applies all technical indicators and returns clean feature DataFrame.
    """
    df = df.copy()

    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_stochastic(df)
    df = add_volume_features(df)
    df = add_price_features(df)
    df = add_calendar_features(df)
    df = create_target(df, horizon=horizon)

    # Drop rows with NaN (from rolling windows)
    df = df.dropna()

    return df


FEATURE_COLUMNS = [
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50",
    "Golden_Cross", "Price_Above_SMA20",
    "RSI", "RSI_Oversold", "RSI_Overbought",
    "MACD", "MACD_Signal", "MACD_Hist", "MACD_Bullish",
    "BB_Width", "BB_Position", "BB_Squeeze",
    "ATR", "ATR_Pct",
    "Stoch_K", "Stoch_D",
    "Volume_Ratio", "Volume_Spike", "OBV",
    "Returns_1d", "Returns_5d", "Returns_20d",
    "Volatility_20d", "High_Low_Range", "Gap",
    "Momentum_10", "Momentum_20",
    "DayOfWeek", "Month", "Quarter", "IsMonday", "IsFriday",
]
