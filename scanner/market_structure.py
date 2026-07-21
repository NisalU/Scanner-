"""
Market Structure Engine
- Detects swing highs/lows
- Classifies HH/HL/LH/LL sequences
- Identifies Break of Structure (BOS) and Change of Character (CHOCH)
- Builds support/resistance/liquidity zones
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ATR_PERIOD, SWING_LOOKBACK, ZONE_MERGE_ATR_MULT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    idx: int
    ts: pd.Timestamp
    price: float
    kind: str   # 'HIGH' | 'LOW'


@dataclass
class Zone:
    kind: str          # 'SUPPORT' | 'RESISTANCE' | 'LIQUIDITY'
    top: float
    bottom: float
    strength: int = 1  # number of touches
    broken: bool = False

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class StructureResult:
    trend: str                      # BULL | BEAR | NEUTRAL | STRONG_BULL | STRONG_BEAR
    last_bos: Optional[str] = None  # BOS_BULL | BOS_BEAR
    last_choch: Optional[str] = None
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows:  List[SwingPoint] = field(default_factory=list)
    support_zones: List[Zone] = field(default_factory=list)
    resistance_zones: List[Zone] = field(default_factory=list)
    liquidity_zones: List[Zone] = field(default_factory=list)
    hh_hl: bool = False   # most recent swings form HH + HL
    lh_ll: bool = False   # most recent swings form LH + LL
    confirmations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Identify swing highs and lows using a rolling pivot approach.
    A swing high is a local max with `lookback` bars on each side.
    """
    highs: List[SwingPoint] = []
    lows:  List[SwingPoint] = []

    n = len(df)
    if n < lookback * 2 + 1:
        return highs, lows

    high_arr  = df["high"].values
    low_arr   = df["low"].values
    index_arr = df.index

    for i in range(lookback, n - lookback):
        window_h = high_arr[i - lookback : i + lookback + 1]
        window_l = low_arr[i - lookback : i + lookback + 1]
        mid_h    = high_arr[i]
        mid_l    = low_arr[i]

        if mid_h == window_h.max():
            highs.append(SwingPoint(i, index_arr[i], float(mid_h), "HIGH"))
        if mid_l == window_l.min():
            lows.append(SwingPoint(i, index_arr[i], float(mid_l), "LOW"))

    return highs, lows


# ---------------------------------------------------------------------------
# Sequence analysis
# ---------------------------------------------------------------------------

def _classify_sequence(points: List[SwingPoint]) -> List[str]:
    """Return a list of labels for consecutive pivot points."""
    if len(points) < 2:
        return []
    labels = []
    for i in range(1, len(points)):
        prev = points[i - 1].price
        curr = points[i].price
        if points[0].kind == "HIGH":
            labels.append("HH" if curr > prev else "LH")
        else:
            labels.append("HL" if curr > prev else "LL")
    return labels


def _detect_bos(
    highs: List[SwingPoint],
    lows:  List[SwingPoint],
    df:    pd.DataFrame,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Break of Structure: price closes beyond the previous swing extreme.
    Change of Character: first opposite-direction BOS after a trend.
    """
    if not highs or not lows:
        return None, None

    last_close = df["close"].iloc[-1]

    bos = None
    # BOS Bull: close breaks above last swing high
    if last_close > highs[-1].price:
        bos = "BOS_BULL"
    # BOS Bear: close breaks below last swing low
    elif last_close < lows[-1].price:
        bos = "BOS_BEAR"

    # CHOCH: price retests broken structure from the other side
    choch = None
    if len(highs) >= 2 and len(lows) >= 2:
        prev_high = highs[-2].price
        prev_low  = lows[-2].price
        curr_high = highs[-1].price
        curr_low  = lows[-1].price
        # Was bearish (LH sequence) but now breaks above prev high → CHOCH Bull
        if curr_high < prev_high and last_close > prev_high:
            choch = "CHOCH_BULL"
        # Was bullish (HL sequence) but now breaks below prev low → CHOCH Bear
        elif curr_low > prev_low and last_close < prev_low:
            choch = "CHOCH_BEAR"

    return bos, choch


# ---------------------------------------------------------------------------
# Zone construction
# ---------------------------------------------------------------------------

def _build_zones(
    highs: List[SwingPoint],
    lows:  List[SwingPoint],
    atr_val: float,
    zone_mult: float = ZONE_MERGE_ATR_MULT,
) -> Tuple[List[Zone], List[Zone], List[Zone]]:
    """
    Construct support, resistance, and liquidity zones from swing points.
    Nearby pivots (within `zone_mult` × ATR) are merged into a single zone.
    """
    half_width = atr_val * 0.2  # zone width = ±20 % of ATR around pivot

    support_zones:    List[Zone] = []
    resistance_zones: List[Zone] = []
    liquidity_zones:  List[Zone] = []

    def _merge_or_add(zones: List[Zone], price: float, kind: str) -> None:
        for z in zones:
            if abs(z.mid - price) <= atr_val * zone_mult:
                z.top    = max(z.top, price + half_width)
                z.bottom = min(z.bottom, price - half_width)
                z.strength += 1
                return
        zones.append(Zone(kind, price + half_width, price - half_width))

    for sp in lows:
        _merge_or_add(support_zones, sp.price, "SUPPORT")

    for sp in highs:
        _merge_or_add(resistance_zones, sp.price, "RESISTANCE")

    # Liquidity zones: swing highs and lows that were NOT broken (resting stops)
    if highs:
        _merge_or_add(liquidity_zones, highs[-1].price, "LIQUIDITY")
    if lows:
        _merge_or_add(liquidity_zones, lows[-1].price, "LIQUIDITY")

    # Sort by strength descending
    support_zones.sort(key=lambda z: z.strength, reverse=True)
    resistance_zones.sort(key=lambda z: z.strength, reverse=True)

    return support_zones, resistance_zones, liquidity_zones


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def analyse_structure(df: pd.DataFrame) -> StructureResult:
    """
    Full market structure analysis for a single timeframe DataFrame.
    `df` must have OHLCV columns and at least ~30 bars.
    """
    result = StructureResult(trend="NEUTRAL")

    if df.empty or len(df) < SWING_LOOKBACK * 4:
        return result

    highs, lows = detect_swings(df, SWING_LOOKBACK)
    result.swing_highs = highs
    result.swing_lows  = lows

    if not highs or not lows:
        return result

    # Sequence labels
    high_labels = _classify_sequence(highs)
    low_labels  = _classify_sequence(lows)

    # Check last 2 swings for HH+HL or LH+LL
    if len(high_labels) >= 1 and len(low_labels) >= 1:
        last_hl = high_labels[-1]
        last_ll = low_labels[-1]
        if last_hl == "HH" and last_ll == "HL":
            result.hh_hl = True
            result.trend = "BULL"
            result.confirmations.append("HH+HL sequence (bullish)")
        elif last_hl == "LH" and last_ll == "LL":
            result.lh_ll = True
            result.trend = "BEAR"
            result.confirmations.append("LH+LL sequence (bearish)")

    # Count consecutive HH+HL for strong trend
    if len(high_labels) >= 3 and len(low_labels) >= 3:
        bull_streak = all(h == "HH" for h in high_labels[-3:]) and all(l == "HL" for l in low_labels[-3:])
        bear_streak = all(h == "LH" for h in high_labels[-3:]) and all(l == "LL" for l in low_labels[-3:])
        if bull_streak:
            result.trend = "STRONG_BULL"
            result.confirmations.append("3× consecutive HH+HL (strong bull)")
        elif bear_streak:
            result.trend = "STRONG_BEAR"
            result.confirmations.append("3× consecutive LH+LL (strong bear)")

    # BOS / CHOCH
    bos, choch = _detect_bos(highs, lows, df)
    result.last_bos   = bos
    result.last_choch = choch
    if bos:
        result.confirmations.append(f"BOS: {bos}")
    if choch:
        result.confirmations.append(f"CHOCH: {choch}")

    # ATR for zone sizing
    atr_col = f"atr_{ATR_PERIOD}"
    if atr_col in df.columns:
        atr_val = float(df[atr_col].iloc[-1])
    else:
        atr_val = float((df["high"] - df["low"]).rolling(ATR_PERIOD).mean().iloc[-1])

    # Zones
    sup, res, liq = _build_zones(highs, lows, atr_val)
    result.support_zones    = sup
    result.resistance_zones = res
    result.liquidity_zones  = liq

    return result


# ---------------------------------------------------------------------------
# Convenience: nearest zone to current price
# ---------------------------------------------------------------------------

def nearest_support(result: StructureResult, price: float) -> Optional[Zone]:
    below = [z for z in result.support_zones if z.top < price]
    return max(below, key=lambda z: z.top) if below else None


def nearest_resistance(result: StructureResult, price: float) -> Optional[Zone]:
    above = [z for z in result.resistance_zones if z.bottom > price]
    return min(above, key=lambda z: z.bottom) if above else None


def price_in_zone(result: StructureResult, price: float) -> Optional[Zone]:
    all_zones = (
        result.support_zones + result.resistance_zones + result.liquidity_zones
    )
    for z in all_zones:
        if z.contains(price):
            return z
    return None
