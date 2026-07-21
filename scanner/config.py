"""
Configuration module — loads environment variables and sets global constants.
All tuneable parameters live here; no magic numbers in other modules.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exchange credentials
# ---------------------------------------------------------------------------

BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH: str = os.getenv("DB_PATH", "scanner.db")

# ---------------------------------------------------------------------------
# Scanner behaviour
# ---------------------------------------------------------------------------

# Timeframes used in multi-TF analysis (ccxt notation)
TIMEFRAMES: List[str] = ["4h", "1h", "15m", "5m", "1m"]

# Minimum USDT 24 h volume to include a pair
MIN_VOLUME_USDT: float = float(os.getenv("MIN_VOLUME_USDT", "10_000_000"))

# How many candles to fetch per timeframe for indicator seeding
CANDLE_LOOKBACK: int = int(os.getenv("CANDLE_LOOKBACK", "300"))

# How many top-volume pairs to actively scan
MAX_PAIRS: int = int(os.getenv("MAX_PAIRS", "100"))

# Seconds between full scanner refresh cycles
SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "60"))

# Minimum composite score to generate an alert
ALERT_SCORE_THRESHOLD: int = int(os.getenv("ALERT_SCORE_THRESHOLD", "80"))

# Cooldown between alerts for the same pair (seconds)
ALERT_COOLDOWN: int = int(os.getenv("ALERT_COOLDOWN", "3600"))

# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------

EMA_FAST: int = 20
EMA_MID: int = 50
EMA_SLOW: int = 200

RSI_PERIOD: int = 14
ATR_PERIOD: int = 14

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

BB_PERIOD: int = 20
BB_STD: float = 2.0

VOLUME_MA_PERIOD: int = 20

# ---------------------------------------------------------------------------
# Volume thresholds
# ---------------------------------------------------------------------------

VOLUME_SPIKE_MULTIPLIER: float = 2.0   # current > avg * this → spike
VOLUME_EXHAUSTION_RATIO: float = 3.0   # climax volume threshold

# ---------------------------------------------------------------------------
# Market structure
# ---------------------------------------------------------------------------

# Minimum swing lookback for pivot detection
SWING_LOOKBACK: int = 5

# Merge S/R zones within this ATR multiple
ZONE_MERGE_ATR_MULT: float = 0.5

# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

ORDER_BOOK_DEPTH: int = 20            # levels to request
WALL_THRESHOLD_MULT: float = 5.0     # qty > avg * this → "wall"
SPOOF_PULL_THRESHOLD: float = 0.8    # 80 % reduction within a tick

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 100)
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    "trend":            20,
    "market_structure": 20,
    "vwap":             15,
    "volume":           15,
    "order_flow":       25,
    "volatility":        5,
}
assert sum(SCORE_WEIGHTS.values()) == 100, "Score weights must sum to 100"

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

BACKTEST_START: str = os.getenv("BACKTEST_START", "2024-01-01")
BACKTEST_END: str   = os.getenv("BACKTEST_END",   "2025-01-01")
BACKTEST_TIMEFRAME: str = os.getenv("BACKTEST_TIMEFRAME", "1h")
RISK_PER_TRADE: float = 0.01     # 1 % of equity per trade
REWARD_RISK_RATIO: float = 2.0   # default TP = SL * this
