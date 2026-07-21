"""
Indicators module — pure-pandas/numpy implementations.
All functions take a DataFrame with columns [open, high, low, close, volume]
and return a new DataFrame with indicator columns appended.
No external TA library required; add pandas_ta as an optional backend.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    EMA_FAST,
    EMA_MID,
    EMA_SLOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    VOLUME_MA_PERIOD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (RMA) used in ATR / RSI."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def ema(df: pd.DataFrame, periods: Tuple[int, ...] = (EMA_FAST, EMA_MID, EMA_SLOW)) -> pd.DataFrame:
    """Add EMA columns: ema_20, ema_50, ema_200 (or custom periods)."""
    df = df.copy()
    for p in periods:
        df[f"ema_{p}"] = _ema(df["close"], p)
    return df


def rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = _rma(gain, period)
    avg_l = _rma(loss, period)
    rs = avg_g / (avg_l + 1e-10)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    df = df.copy()
    hl   = df["high"] - df["low"]
    hpc  = (df["high"] - df["close"].shift(1)).abs()
    lpc  = (df["low"]  - df["close"].shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df[f"atr_{period}"] = _rma(tr, period)
    return df


def macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    df = df.copy()
    fast_ema = _ema(df["close"], fast)
    slow_ema = _ema(df["close"], slow)
    df["macd_line"]   = fast_ema - slow_ema
    df["macd_signal"] = _ema(df["macd_line"], signal)
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]
    return df


def bollinger_bands(
    df: pd.DataFrame, period: int = BB_PERIOD, std: float = BB_STD
) -> pd.DataFrame:
    df = df.copy()
    ma     = df["close"].rolling(period).mean()
    stddev = df["close"].rolling(period).std()
    df["bb_mid"]   = ma
    df["bb_upper"] = ma + std * stddev
    df["bb_lower"] = ma - std * stddev
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)
    df["bb_pct"]   = (df["close"] - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"] + 1e-10
    )
    return df


def volume_ma(df: pd.DataFrame, period: int = VOLUME_MA_PERIOD) -> pd.DataFrame:
    df = df.copy()
    df[f"vol_ma_{period}"] = df["volume"].rolling(period).mean()
    df["vol_ratio"]        = df["volume"] / (df[f"vol_ma_{period}"] + 1e-10)
    return df


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anchored VWAP — resets at the start of each trading day (midnight UTC).
    Works correctly on any intraday timeframe.
    """
    df = df.copy()

    if df.empty:
        df["vwap"] = np.nan
        return df

    # Ensure UTC datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    tpv     = typical * df["volume"]

    # Group by date to anchor each day
    dates = df.index.normalize()
    df["vwap"] = (
        tpv.groupby(dates).cumsum() /
        df["volume"].groupby(dates).cumsum()
    )
    return df


def stochastic_rsi(
    df: pd.DataFrame, rsi_len: int = RSI_PERIOD, stoch_len: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> pd.DataFrame:
    """Stochastic RSI — useful for overbought/oversold confluence."""
    df = df.copy()
    df = rsi(df, rsi_len)
    rsi_col = df[f"rsi_{rsi_len}"]
    lo  = rsi_col.rolling(stoch_len).min()
    hi  = rsi_col.rolling(stoch_len).max()
    raw = (rsi_col - lo) / (hi - lo + 1e-10) * 100
    df["stoch_rsi_k"] = raw.rolling(smooth_k).mean()
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(smooth_d).mean()
    return df


# ---------------------------------------------------------------------------
# Composite: apply all indicators at once
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator to `df` and return the enriched DataFrame."""
    if df.empty or len(df) < EMA_SLOW:
        logger.debug("Not enough candles for full indicator suite (have %d)", len(df))
        return df

    df = ema(df)
    df = rsi(df)
    df = atr(df)
    df = macd(df)
    df = bollinger_bands(df)
    df = volume_ma(df)
    df = vwap(df)
    return df


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------

def ema_trend(df: pd.DataFrame) -> str:
    """
    Classify trend from EMA stack on last closed candle.
    Returns: 'STRONG_BULL' | 'BULL' | 'NEUTRAL' | 'BEAR' | 'STRONG_BEAR'
    """
    last = df.iloc[-1]
    cols = [f"ema_{EMA_FAST}", f"ema_{EMA_MID}", f"ema_{EMA_SLOW}"]
    if not all(c in last.index for c in cols):
        return "NEUTRAL"

    e20, e50, e200 = last[cols[0]], last[cols[1]], last[cols[2]]
    close = last["close"]

    if close > e20 > e50 > e200:
        return "STRONG_BULL"
    if close > e50 > e200:
        return "BULL"
    if close < e20 < e50 < e200:
        return "STRONG_BEAR"
    if close < e50 < e200:
        return "BEAR"
    return "NEUTRAL"


def vwap_position(df: pd.DataFrame) -> str:
    """Returns 'ABOVE' | 'BELOW' | 'AT' relative to VWAP."""
    if "vwap" not in df.columns:
        return "UNKNOWN"
    last = df.iloc[-1]
    diff_pct = (last["close"] - last["vwap"]) / (last["vwap"] + 1e-10) * 100
    if diff_pct > 0.1:
        return "ABOVE"
    if diff_pct < -0.1:
        return "BELOW"
    return "AT"


def rsi_zone(df: pd.DataFrame) -> str:
    """Returns 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL'."""
    col = f"rsi_{RSI_PERIOD}"
    if col not in df.columns:
        return "NEUTRAL"
    val = df[col].iloc[-1]
    if val > 70:
        return "OVERBOUGHT"
    if val < 30:
        return "OVERSOLD"
    return "NEUTRAL"


def macd_signal_cross(df: pd.DataFrame) -> str:
    """Detects MACD line crossing signal line on last two bars."""
    if "macd_line" not in df.columns or len(df) < 2:
        return "NONE"
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    if prev["macd_line"] < prev["macd_signal"] and curr["macd_line"] > curr["macd_signal"]:
        return "BULL_CROSS"
    if prev["macd_line"] > prev["macd_signal"] and curr["macd_line"] < curr["macd_signal"]:
        return "BEAR_CROSS"
    return "NONE"


def bb_squeeze(df: pd.DataFrame, lookback: int = 20) -> bool:
    """True when BB width is at its tightest in `lookback` bars — volatility squeeze."""
    if "bb_width" not in df.columns or len(df) < lookback:
        return False
    recent = df["bb_width"].iloc[-lookback:]
    return float(df["bb_width"].iloc[-1]) == float(recent.min())


def volume_spike(df: pd.DataFrame, multiplier: float = 2.0) -> bool:
    """True when last bar volume exceeds `multiplier` × rolling average."""
    if "vol_ratio" not in df.columns:
        return False
    return float(df["vol_ratio"].iloc[-1]) >= multiplier


def atr_expansion(df: pd.DataFrame, lookback: int = 14) -> bool:
    """True when ATR is expanding (last value > rolling mean of ATR)."""
    col = f"atr_{ATR_PERIOD}"
    if col not in df.columns or len(df) < lookback:
        return False
    avg_atr = df[col].iloc[-lookback:].mean()
    return float(df[col].iloc[-1]) > avg_atr
