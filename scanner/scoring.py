"""
Signal Scoring Engine
Produces a 0-100 composite score from six weighted dimensions.
Only signals above ALERT_SCORE_THRESHOLD are promoted to alerts.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    ALERT_SCORE_THRESHOLD,
    ATR_PERIOD,
    SCORE_WEIGHTS,
    VOLUME_SPIKE_MULTIPLIER,
)
from .indicators import (
    atr_expansion,
    ema_trend,
    macd_signal_cross,
    rsi_zone,
    vwap_position,
    volume_spike,
)
from .market_structure import StructureResult, nearest_resistance, nearest_support
from .orderflow import OrderFlowResult
from .liquidity import OrderBookResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    trend:            float = 0.0
    market_structure: float = 0.0
    vwap:             float = 0.0
    volume:           float = 0.0
    order_flow:       float = 0.0
    volatility:       float = 0.0

    @property
    def total(self) -> float:
        return (
            self.trend
            + self.market_structure
            + self.vwap
            + self.volume
            + self.order_flow
            + self.volatility
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "trend":            round(self.trend, 1),
            "market_structure": round(self.market_structure, 1),
            "vwap":             round(self.vwap, 1),
            "volume":           round(self.volume, 1),
            "order_flow":       round(self.order_flow, 1),
            "volatility":       round(self.volatility, 1),
            "total":            round(self.total, 1),
        }


@dataclass
class SignalResult:
    symbol:        str
    direction:     str          # LONG | SHORT
    score:         float
    breakdown:     ScoreBreakdown
    confirmations: List[str]    = field(default_factory=list)
    entry_price:   float = 0.0
    tp_price:      float = 0.0
    sl_price:      float = 0.0
    is_valid:      bool  = False

    @property
    def alert_worthy(self) -> bool:
        return self.score >= ALERT_SCORE_THRESHOLD


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

def _score_trend(
    df_4h: pd.DataFrame, df_1h: pd.DataFrame, direction: str
) -> Tuple[float, List[str]]:
    """Up to SCORE_WEIGHTS['trend'] points."""
    max_pts = SCORE_WEIGHTS["trend"]
    pts = 0.0
    conf: List[str] = []

    trend_4h = ema_trend(df_4h) if not df_4h.empty else "NEUTRAL"
    trend_1h = ema_trend(df_1h) if not df_1h.empty else "NEUTRAL"

    bull_states = {"BULL", "STRONG_BULL"}
    bear_states = {"BEAR", "STRONG_BEAR"}

    if direction == "LONG":
        if trend_4h in bull_states:
            pts += max_pts * 0.6
            conf.append(f"4H {trend_4h} trend (EMA stack)")
        elif trend_4h == "NEUTRAL":
            pts += max_pts * 0.2

        if trend_1h in bull_states:
            pts += max_pts * 0.4
            conf.append(f"1H {trend_1h} trend (EMA stack)")
        elif trend_1h == "NEUTRAL":
            pts += max_pts * 0.1

    else:  # SHORT
        if trend_4h in bear_states:
            pts += max_pts * 0.6
            conf.append(f"4H {trend_4h} trend (EMA stack)")
        elif trend_4h == "NEUTRAL":
            pts += max_pts * 0.2

        if trend_1h in bear_states:
            pts += max_pts * 0.4
            conf.append(f"1H {trend_1h} trend (EMA stack)")
        elif trend_1h == "NEUTRAL":
            pts += max_pts * 0.1

    return min(pts, max_pts), conf


def _score_market_structure(
    struct_4h: StructureResult,
    struct_1h: StructureResult,
    direction:  str,
    price:      float,
) -> Tuple[float, List[str]]:
    """Up to SCORE_WEIGHTS['market_structure'] points."""
    max_pts = SCORE_WEIGHTS["market_structure"]
    pts = 0.0
    conf: List[str] = []

    bull_trends = {"BULL", "STRONG_BULL"}
    bear_trends = {"BEAR", "STRONG_BEAR"}

    if direction == "LONG":
        if struct_4h.trend in bull_trends:
            pts += max_pts * 0.35
        if struct_4h.hh_hl:
            pts += max_pts * 0.2
            conf.append("4H HH+HL sequence")
        if struct_1h.hh_hl:
            pts += max_pts * 0.2
            conf.append("1H HH+HL sequence")
        if struct_1h.last_bos == "BOS_BULL":
            pts += max_pts * 0.15
            conf.append("1H Break of Structure (bullish)")
        if struct_1h.last_choch == "CHOCH_BULL":
            pts += max_pts * 0.1
            conf.append("1H Change of Character (bullish)")
        sup = nearest_support(struct_1h, price)
        if sup and sup.strength >= 2:
            pts += max_pts * 0.1
            conf.append(f"Strong support zone @ {sup.mid:.4f} ({sup.strength} touches)")
    else:  # SHORT
        if struct_4h.trend in bear_trends:
            pts += max_pts * 0.35
        if struct_4h.lh_ll:
            pts += max_pts * 0.2
            conf.append("4H LH+LL sequence")
        if struct_1h.lh_ll:
            pts += max_pts * 0.2
            conf.append("1H LH+LL sequence")
        if struct_1h.last_bos == "BOS_BEAR":
            pts += max_pts * 0.15
            conf.append("1H Break of Structure (bearish)")
        if struct_1h.last_choch == "CHOCH_BEAR":
            pts += max_pts * 0.1
            conf.append("1H Change of Character (bearish)")
        res = nearest_resistance(struct_1h, price)
        if res and res.strength >= 2:
            pts += max_pts * 0.1
            conf.append(f"Strong resistance zone @ {res.mid:.4f} ({res.strength} touches)")

    return min(pts, max_pts), conf


def _score_vwap(
    df_1h: pd.DataFrame, direction: str
) -> Tuple[float, List[str]]:
    """Up to SCORE_WEIGHTS['vwap'] points."""
    max_pts = SCORE_WEIGHTS["vwap"]
    pts = 0.0
    conf: List[str] = []

    vp = vwap_position(df_1h)
    macd_x = macd_signal_cross(df_1h)
    rsi_z  = rsi_zone(df_1h)

    if direction == "LONG":
        if vp == "ABOVE":
            pts += max_pts * 0.5
            conf.append("Price above 1H VWAP")
        if macd_x == "BULL_CROSS":
            pts += max_pts * 0.3
            conf.append("1H MACD bullish cross")
        if rsi_z == "OVERSOLD":
            pts += max_pts * 0.2
            conf.append("RSI oversold bounce")
        elif rsi_z == "NEUTRAL":
            pts += max_pts * 0.1
    else:
        if vp == "BELOW":
            pts += max_pts * 0.5
            conf.append("Price below 1H VWAP")
        if macd_x == "BEAR_CROSS":
            pts += max_pts * 0.3
            conf.append("1H MACD bearish cross")
        if rsi_z == "OVERBOUGHT":
            pts += max_pts * 0.2
            conf.append("RSI overbought rejection")
        elif rsi_z == "NEUTRAL":
            pts += max_pts * 0.1

    return min(pts, max_pts), conf


def _score_volume(
    df_15m: pd.DataFrame, direction: str
) -> Tuple[float, List[str]]:
    """Up to SCORE_WEIGHTS['volume'] points."""
    max_pts = SCORE_WEIGHTS["volume"]
    pts = 0.0
    conf: List[str] = []

    if df_15m.empty or "vol_ratio" not in df_15m.columns:
        return 0.0, []

    vol_ratio = float(df_15m["vol_ratio"].iloc[-1])

    if vol_ratio >= VOLUME_SPIKE_MULTIPLIER:
        pts += max_pts * 0.7
        conf.append(f"Volume spike: {vol_ratio:.1f}× average")
    elif vol_ratio >= 1.5:
        pts += max_pts * 0.4
        conf.append(f"Above-average volume: {vol_ratio:.1f}× average")

    # Breakout volume: rising closes on expanding volume
    if len(df_15m) >= 3:
        recent_vols   = df_15m["volume"].iloc[-3:].values
        recent_closes = df_15m["close"].iloc[-3:].values
        if direction == "LONG":
            if all(recent_vols[i] >= recent_vols[i-1] for i in range(1, 3)):
                if all(recent_closes[i] >= recent_closes[i-1] for i in range(1, 3)):
                    pts += max_pts * 0.3
                    conf.append("Breakout volume: expanding volume on rising closes")
        else:
            if all(recent_vols[i] >= recent_vols[i-1] for i in range(1, 3)):
                if all(recent_closes[i] <= recent_closes[i-1] for i in range(1, 3)):
                    pts += max_pts * 0.3
                    conf.append("Breakdown volume: expanding volume on falling closes")

    return min(pts, max_pts), conf


def _score_order_flow(
    of_result: Optional[OrderFlowResult],
    ob_result: Optional[OrderBookResult],
    direction:  str,
) -> Tuple[float, List[str]]:
    """Up to SCORE_WEIGHTS['order_flow'] points."""
    max_pts = SCORE_WEIGHTS["order_flow"]
    pts = 0.0
    conf: List[str] = []

    if of_result:
        if direction == "LONG":
            if of_result.is_bullish_absorption:
                pts += max_pts * 0.4
                conf.append("Bullish absorption detected")
            if of_result.is_delta_divergence:
                bull_dd = any("Bullish" in c for c in of_result.confirmations)
                if bull_dd:
                    pts += max_pts * 0.25
                    conf.append("Bullish delta divergence")
            if of_result.current.delta_pct > 30:
                pts += max_pts * 0.15
                conf.append(f"Positive delta: {of_result.current.delta_pct:.0f}%")
        else:
            if of_result.is_bearish_absorption:
                pts += max_pts * 0.4
                conf.append("Bearish absorption detected")
            if of_result.is_delta_divergence:
                bear_dd = any("Bearish" in c for c in of_result.confirmations)
                if bear_dd:
                    pts += max_pts * 0.25
                    conf.append("Bearish delta divergence")
            if of_result.current.delta_pct < -30:
                pts += max_pts * 0.15
                conf.append(f"Negative delta: {of_result.current.delta_pct:.0f}%")

    if ob_result:
        if direction == "LONG" and ob_result.bias == "BID_HEAVY":
            pts += max_pts * 0.15
            conf.append("Order book bid-heavy")
        elif direction == "SHORT" and ob_result.bias == "ASK_HEAVY":
            pts += max_pts * 0.15
            conf.append("Order book ask-heavy")

        # Wall acts as confirmation
        if direction == "LONG" and any(w.side == "BID" for w in ob_result.walls):
            pts += max_pts * 0.1
            conf.append("Large bid wall (support)")
        elif direction == "SHORT" and any(w.side == "ASK" for w in ob_result.walls):
            pts += max_pts * 0.1
            conf.append("Large ask wall (resistance)")

    return min(pts, max_pts), conf


def _score_volatility(df_15m: pd.DataFrame) -> Tuple[float, List[str]]:
    """Up to SCORE_WEIGHTS['volatility'] points — favours volatility expansion."""
    max_pts = SCORE_WEIGHTS["volatility"]
    conf: List[str] = []
    pts = 0.0

    if df_15m.empty:
        return 0.0, []

    if atr_expansion(df_15m):
        pts += max_pts * 0.7
        conf.append("ATR expanding (volatility breakout)")

    if "bb_width" in df_15m.columns and len(df_15m) >= 20:
        bb_now = float(df_15m["bb_width"].iloc[-1])
        bb_avg = float(df_15m["bb_width"].iloc[-20:].mean())
        if bb_now > bb_avg * 1.2:
            pts += max_pts * 0.3
            conf.append("Bollinger Bands expanding")

    return min(pts, max_pts), conf


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_signal(
    symbol:     str,
    direction:  str,
    price:      float,
    df_4h:      pd.DataFrame,
    df_1h:      pd.DataFrame,
    df_15m:     pd.DataFrame,
    struct_4h:  StructureResult,
    struct_1h:  StructureResult,
    of_result:  Optional[OrderFlowResult] = None,
    ob_result:  Optional[OrderBookResult] = None,
) -> SignalResult:
    """
    Compute the composite score for a LONG or SHORT signal.
    Returns a SignalResult; check .alert_worthy to decide whether to alert.
    """
    bd = ScoreBreakdown()
    all_conf: List[str] = []

    bd.trend, c = _score_trend(df_4h, df_1h, direction)
    all_conf.extend(c)

    bd.market_structure, c = _score_market_structure(struct_4h, struct_1h, direction, price)
    all_conf.extend(c)

    bd.vwap, c = _score_vwap(df_1h, direction)
    all_conf.extend(c)

    bd.volume, c = _score_volume(df_15m, direction)
    all_conf.extend(c)

    bd.order_flow, c = _score_order_flow(of_result, ob_result, direction)
    all_conf.extend(c)

    bd.volatility, c = _score_volatility(df_15m)
    all_conf.extend(c)

    total_score = min(bd.total, 100.0)

    # ------------------------------------------------------------------
    # Entry, TP, SL levels
    # ------------------------------------------------------------------
    atr_col = f"atr_{ATR_PERIOD}"
    atr_val = float(df_15m[atr_col].iloc[-1]) if (atr_col in df_15m.columns and not df_15m.empty) else price * 0.002

    entry = price
    if direction == "LONG":
        sl  = entry - atr_val * 1.5
        tp  = entry + atr_val * 3.0
    else:
        sl  = entry + atr_val * 1.5
        tp  = entry - atr_val * 3.0

    # Override SL with nearest structure if available
    if direction == "LONG":
        sup = nearest_support(struct_1h, price)
        if sup and sup.bottom > sl:
            sl = sup.bottom * 0.999
            all_conf.append(f"SL below support zone @ {sl:.4f}")
    else:
        res = nearest_resistance(struct_1h, price)
        if res and res.top < sl:
            sl = res.top * 1.001
            all_conf.append(f"SL above resistance zone @ {sl:.4f}")

    return SignalResult(
        symbol=symbol,
        direction=direction,
        score=round(total_score, 1),
        breakdown=bd,
        confirmations=all_conf,
        entry_price=round(entry, 6),
        tp_price=round(tp, 6),
        sl_price=round(sl, 6),
        is_valid=total_score >= ALERT_SCORE_THRESHOLD,
    )


def determine_direction(
    struct_4h: StructureResult,
    struct_1h: StructureResult,
    df_1h:     pd.DataFrame,
) -> Optional[str]:
    """
    Heuristic to decide whether to score LONG, SHORT, or skip.
    Returns 'LONG' | 'SHORT' | None.
    """
    bull_trends = {"BULL", "STRONG_BULL"}
    bear_trends = {"BEAR", "STRONG_BEAR"}

    # Both timeframes must agree
    if struct_4h.trend in bull_trends and struct_1h.trend in bull_trends:
        return "LONG"
    if struct_4h.trend in bear_trends and struct_1h.trend in bear_trends:
        return "SHORT"

    # BOS direction
    if struct_1h.last_bos == "BOS_BULL":
        return "LONG"
    if struct_1h.last_bos == "BOS_BEAR":
        return "SHORT"

    # VWAP fallback
    vp = vwap_position(df_1h)
    if vp == "ABOVE":
        return "LONG"
    if vp == "BELOW":
        return "SHORT"

    return None
