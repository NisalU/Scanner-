"""
Order Flow Engine
- Aggregates buy/sell volume from real-time trades
- Calculates delta and delta percentage
- Detects imbalance, absorption, and climax volume events
- Provides footprint-style analysis per candle
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import VOLUME_SPIKE_MULTIPLIER, VOLUME_EXHAUSTION_RATIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandleFlow:
    """Order-flow data for a single candle interval."""
    ts: int                  # open time ms
    buy_vol:   float = 0.0
    sell_vol:  float = 0.0
    trades:    int   = 0
    avg_price: float = 0.0

    @property
    def total_vol(self) -> float:
        return self.buy_vol + self.sell_vol + 1e-10

    @property
    def delta(self) -> float:
        return self.buy_vol - self.sell_vol

    @property
    def delta_pct(self) -> float:
        return self.delta / self.total_vol * 100

    @property
    def buy_pct(self) -> float:
        return self.buy_vol / self.total_vol * 100

    @property
    def sell_pct(self) -> float:
        return self.sell_vol / self.total_vol * 100


@dataclass
class OrderFlowResult:
    symbol: str
    current:    CandleFlow = field(default_factory=lambda: CandleFlow(0))
    history:    List[CandleFlow] = field(default_factory=list)
    # Detected events
    is_bullish_absorption: bool = False
    is_bearish_absorption: bool = False
    is_imbalance:          bool = False
    is_delta_divergence:   bool = False
    is_climax_volume:      bool = False
    event_type:            str  = "NONE"
    confirmations:         List[str] = field(default_factory=list)
    score_component:       float = 0.0  # 0-25 contribution


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class OrderFlowAggregator:
    """
    Aggregates raw trades into candle-level buy/sell volumes.
    Maintains a rolling history of `max_history` candles.
    """

    def __init__(self, candle_seconds: int = 60, max_history: int = 100) -> None:
        self._candle_ms  = candle_seconds * 1000
        self._max_history = max_history
        # symbol → list of CandleFlow
        self._candles: Dict[str, List[CandleFlow]] = {}
        # symbol → current open candle
        self._current: Dict[str, CandleFlow] = {}

    def process_trade(
        self,
        symbol: str,
        ts_ms: int,
        price: float,
        qty: float,
        side: str,          # 'buy' | 'sell'
    ) -> None:
        candle_open = (ts_ms // self._candle_ms) * self._candle_ms

        if symbol not in self._current:
            self._current[symbol] = CandleFlow(ts=candle_open)
            self._candles[symbol] = []

        cur = self._current[symbol]

        # Candle rollover
        if candle_open > cur.ts:
            self._candles[symbol].append(cur)
            if len(self._candles[symbol]) > self._max_history:
                self._candles[symbol].pop(0)
            cur = CandleFlow(ts=candle_open)
            self._current[symbol] = cur

        # Accumulate
        if side == "buy":
            cur.buy_vol += qty
        else:
            cur.sell_vol += qty
        cur.trades += 1
        cur.avg_price = (cur.avg_price * (cur.trades - 1) + price) / cur.trades

    def process_trades_batch(self, symbol: str, trades: List[Dict]) -> None:
        """Process a list of trade dicts (from WS store or REST)."""
        for t in trades:
            self.process_trade(
                symbol,
                int(t.get("ts", t.get("timestamp", 0))),
                float(t["price"]),
                float(t.get("qty", t.get("amount", 0))),
                str(t.get("side", "buy")).lower(),
            )

    def get_candles(self, symbol: str) -> List[CandleFlow]:
        hist = list(self._candles.get(symbol, []))
        cur  = self._current.get(symbol)
        if cur:
            hist = hist + [cur]
        return hist

    def get_current(self, symbol: str) -> Optional[CandleFlow]:
        return self._current.get(symbol)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _avg_delta(candles: List[CandleFlow], n: int = 10) -> float:
    recent = candles[-n:]
    if not recent:
        return 0.0
    return np.mean([c.delta_pct for c in recent])


def _avg_volume(candles: List[CandleFlow], n: int = 20) -> float:
    recent = candles[-n:]
    if not recent:
        return 0.0
    return np.mean([c.total_vol for c in recent])


def detect_bullish_absorption(
    candles: List[CandleFlow],
    df_close: pd.Series,
) -> Tuple[bool, str]:
    """
    Bullish absorption:
    - High sell volume (price tried to drop)
    - Price held / support zone held
    - Delta improved in last 2-3 candles
    """
    if len(candles) < 5:
        return False, ""

    recent = candles[-5:]
    sell_pressure = [c for c in recent if c.sell_pct > 60]
    if len(sell_pressure) < 2:
        return False, ""

    # Delta improving: last candle's delta better than 3 bars ago
    delta_now  = recent[-1].delta_pct
    delta_then = recent[-3].delta_pct
    delta_improving = delta_now > delta_then

    # Price holding (not making new lows)
    if len(df_close) >= 5:
        price_holding = df_close.iloc[-1] >= df_close.iloc[-5] * 0.999
    else:
        price_holding = True

    if delta_improving and price_holding:
        return True, "Bullish absorption: high sell pressure, price holding, delta recovering"
    return False, ""


def detect_bearish_absorption(
    candles: List[CandleFlow],
    df_close: pd.Series,
) -> Tuple[bool, str]:
    """
    Bearish absorption:
    - High buy volume (price tried to push up)
    - Price rejected resistance
    - Delta weakened in last 2-3 candles
    """
    if len(candles) < 5:
        return False, ""

    recent = candles[-5:]
    buy_pressure = [c for c in recent if c.buy_pct > 60]
    if len(buy_pressure) < 2:
        return False, ""

    delta_now  = recent[-1].delta_pct
    delta_then = recent[-3].delta_pct
    delta_weakening = delta_now < delta_then

    if len(df_close) >= 5:
        price_failing = df_close.iloc[-1] <= df_close.iloc[-5] * 1.001
    else:
        price_failing = True

    if delta_weakening and price_failing:
        return True, "Bearish absorption: high buy pressure, price rejected, delta weakening"
    return False, ""


def detect_imbalance(candles: List[CandleFlow], threshold: float = 70.0) -> Tuple[bool, str]:
    """
    Imbalance: latest candle is strongly one-sided (> threshold % on one side).
    """
    if not candles:
        return False, ""
    c = candles[-1]
    if c.buy_pct > threshold:
        return True, f"Bullish imbalance: {c.buy_pct:.0f}% buy volume"
    if c.sell_pct > threshold:
        return True, f"Bearish imbalance: {c.sell_pct:.0f}% sell volume"
    return False, ""


def detect_delta_divergence(
    candles: List[CandleFlow], df_close: pd.Series, lookback: int = 5
) -> Tuple[bool, str]:
    """
    Delta divergence: price moves higher but delta declines (hidden weakness) or vice versa.
    """
    if len(candles) < lookback or len(df_close) < lookback:
        return False, ""

    price_up  = df_close.iloc[-1] > df_close.iloc[-lookback]
    delta_down = candles[-1].delta_pct < candles[-lookback].delta_pct

    if price_up and delta_down:
        return True, "Bearish delta divergence: price rising, delta declining"

    price_down = df_close.iloc[-1] < df_close.iloc[-lookback]
    delta_up   = candles[-1].delta_pct > candles[-lookback].delta_pct

    if price_down and delta_up:
        return True, "Bullish delta divergence: price falling, delta rising"

    return False, ""


def detect_climax_volume(candles: List[CandleFlow]) -> Tuple[bool, str]:
    """Volume exhaustion / climax: volume >> average AND delta doesn't follow."""
    if len(candles) < 20:
        return False, ""
    avg_vol = _avg_volume(candles[:-1], 20)
    cur_vol = candles[-1].total_vol
    if cur_vol > avg_vol * VOLUME_EXHAUSTION_RATIO:
        delta_pct = candles[-1].delta_pct
        if abs(delta_pct) < 20:
            return True, f"Climax volume: {cur_vol/avg_vol:.1f}× avg, delta neutral ({delta_pct:.0f}%)"
    return False, ""


def detect_volume_acceleration(candles: List[CandleFlow], n: int = 5) -> Tuple[bool, str]:
    """Volume consistently accelerating over last n candles."""
    if len(candles) < n:
        return False, ""
    vols = [c.total_vol for c in candles[-n:]]
    increasing = all(vols[i] >= vols[i-1] for i in range(1, len(vols)))
    if increasing and vols[-1] > vols[0] * 1.5:
        return True, f"Volume acceleration: {vols[-1]/vols[0]:.1f}× over {n} bars"
    return False, ""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_orderflow(
    symbol: str,
    candles: List[CandleFlow],
    df: pd.DataFrame,
) -> OrderFlowResult:
    """
    Run all order-flow detectors and return a scored OrderFlowResult.
    `df` should be the 5m or 1m OHLCV DataFrame for price context.
    """
    result = OrderFlowResult(symbol=symbol, history=candles)

    if len(candles) < 5 or df.empty:
        return result

    result.current = candles[-1]

    close_series = df["close"]

    # Detection
    bull_abs, bull_msg = detect_bullish_absorption(candles, close_series)
    bear_abs, bear_msg = detect_bearish_absorption(candles, close_series)
    imbal, imbal_msg   = detect_imbalance(candles)
    delta_div, dd_msg  = detect_delta_divergence(candles, close_series)
    climax, climax_msg = detect_climax_volume(candles)
    vol_acc, vacc_msg  = detect_volume_acceleration(candles)

    result.is_bullish_absorption = bull_abs
    result.is_bearish_absorption = bear_abs
    result.is_imbalance          = imbal
    result.is_delta_divergence   = delta_div
    result.is_climax_volume      = climax

    for msg in [bull_msg, bear_msg, imbal_msg, dd_msg, climax_msg, vacc_msg]:
        if msg:
            result.confirmations.append(msg)

    # Assign primary event type
    if bull_abs:
        result.event_type = "ABSORPTION_BULL"
    elif bear_abs:
        result.event_type = "ABSORPTION_BEAR"
    elif imbal:
        result.event_type = "IMBALANCE"
    elif climax:
        result.event_type = "CLIMAX"
    elif delta_div:
        result.event_type = "DELTA_DIV"

    # Score contribution (0–25)
    score = 0.0
    if bull_abs:                           score += 12
    if bear_abs:                           score += 12
    if imbal:                              score += 5
    if delta_div:                          score += 5
    if climax:                             score += 3
    if vol_acc:                            score += 4
    # Delta on last candle strongly positive → bull
    if result.current.delta_pct > 40:      score += 3
    elif result.current.delta_pct < -40:   score += 3  # bear

    result.score_component = min(score, 25)
    return result
