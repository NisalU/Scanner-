"""
Liquidity & Order Book Analysis
- Identifies large order walls (icebergs / stops)
- Detects liquidity clusters
- Flags possible spoofing (large orders appearing and vanishing)
- Calculates bid/ask imbalance
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ORDER_BOOK_DEPTH, WALL_THRESHOLD_MULT, SPOOF_PULL_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderWall:
    side:      str     # 'BID' | 'ASK'
    price:     float
    qty:       float
    ratio:     float   # qty / avg_qty at that side


@dataclass
class LiquidityCluster:
    side:      str
    price_lo:  float
    price_hi:  float
    total_qty: float
    num_levels: int


@dataclass
class SpoofAlert:
    side:      str
    price:     float
    prev_qty:  float
    curr_qty:  float
    pull_pct:  float   # how much was pulled


@dataclass
class OrderBookResult:
    symbol:          str
    bid_liquidity:   float = 0.0      # total bid qty in book
    ask_liquidity:   float = 0.0      # total ask qty in book
    bid_ask_ratio:   float = 0.0      # bid / ask qty ratio
    spread:          float = 0.0      # ask1 - bid1
    spread_pct:      float = 0.0      # spread / mid * 100
    mid_price:       float = 0.0
    walls:           List[OrderWall]         = field(default_factory=list)
    clusters:        List[LiquidityCluster]  = field(default_factory=list)
    spoof_alerts:    List[SpoofAlert]        = field(default_factory=list)
    bias:            str = "NEUTRAL"  # BID_HEAVY | ASK_HEAVY | NEUTRAL
    confirmations:   List[str] = field(default_factory=list)
    score_component: float = 0.0      # used in scoring


# ---------------------------------------------------------------------------
# Snapshot store for spoof detection (price → qty history)
# ---------------------------------------------------------------------------

class BookHistory:
    """Keeps a short history of order book snapshots per symbol."""

    def __init__(self, max_snapshots: int = 10) -> None:
        self._max = max_snapshots
        # symbol → [{"bids": {price:qty}, "asks": {price:qty}}, ...]
        self._history: Dict[str, List[Dict]] = {}

    def record(self, symbol: str, bids: Dict[float, float], asks: Dict[float, float]) -> None:
        buf = self._history.setdefault(symbol, [])
        buf.append({"bids": dict(bids), "asks": dict(asks)})
        if len(buf) > self._max:
            buf.pop(0)

    def get_prev(self, symbol: str) -> Optional[Dict]:
        buf = self._history.get(symbol, [])
        return buf[-2] if len(buf) >= 2 else None


_book_history = BookHistory()


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _detect_walls(
    levels: List[Tuple[float, float]],
    side: str,
    threshold_mult: float = WALL_THRESHOLD_MULT,
) -> List[OrderWall]:
    """Flag levels whose qty is `threshold_mult` × the average level qty."""
    if not levels:
        return []
    qtys = [q for _, q in levels]
    avg  = np.mean(qtys) + 1e-10
    walls: List[OrderWall] = []
    for price, qty in levels:
        if qty >= avg * threshold_mult:
            walls.append(OrderWall(side=side, price=price, qty=qty, ratio=qty / avg))
    return sorted(walls, key=lambda w: w.ratio, reverse=True)


def _detect_clusters(
    levels: List[Tuple[float, float]],
    side: str,
    cluster_pct: float = 0.002,  # group levels within 0.2% of each other
) -> List[LiquidityCluster]:
    """Group nearby price levels into liquidity clusters."""
    if not levels:
        return []

    clusters: List[LiquidityCluster] = []
    start_price, start_qty = levels[0]
    cluster_qty = start_qty
    cluster_lo  = start_price
    cluster_hi  = start_price
    count = 1

    for price, qty in levels[1:]:
        if abs(price - cluster_hi) / (cluster_hi + 1e-10) <= cluster_pct:
            cluster_qty += qty
            cluster_hi   = max(cluster_hi, price)
            cluster_lo   = min(cluster_lo, price)
            count += 1
        else:
            if count >= 3:
                clusters.append(
                    LiquidityCluster(side, cluster_lo, cluster_hi, cluster_qty, count)
                )
            cluster_lo  = price
            cluster_hi  = price
            cluster_qty = qty
            count = 1

    if count >= 3:
        clusters.append(
            LiquidityCluster(side, cluster_lo, cluster_hi, cluster_qty, count)
        )

    return sorted(clusters, key=lambda c: c.total_qty, reverse=True)


def _detect_spoofing(
    symbol: str,
    bids: Dict[float, float],
    asks: Dict[float, float],
    pull_threshold: float = SPOOF_PULL_THRESHOLD,
) -> List[SpoofAlert]:
    """
    Compare current book to the previous snapshot.
    If a large order shrinks by `pull_threshold` fraction in one tick → possible spoof.
    """
    prev = _book_history.get_prev(symbol)
    if not prev:
        return []

    alerts: List[SpoofAlert] = []

    for side_name, curr_side, prev_side in [
        ("BID", bids, prev["bids"]),
        ("ASK", asks, prev["asks"]),
    ]:
        all_qtys = list(curr_side.values()) + list(prev_side.values())
        if not all_qtys:
            continue
        avg_qty = np.mean(all_qtys) + 1e-10

        for price, prev_qty in prev_side.items():
            if prev_qty < avg_qty * 3:
                continue
            curr_qty = curr_side.get(price, 0.0)
            pull_pct = (prev_qty - curr_qty) / (prev_qty + 1e-10)
            if pull_pct >= pull_threshold:
                alerts.append(
                    SpoofAlert(side_name, price, prev_qty, curr_qty, pull_pct)
                )

    return alerts


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_order_book(
    symbol: str,
    raw_bids: List[List[float]],  # [[price, qty], ...]
    raw_asks: List[List[float]],
) -> OrderBookResult:
    """
    Full order-book analysis.
    `raw_bids` and `raw_asks` are as returned by ccxt.fetch_order_book.
    """
    result = OrderBookResult(symbol=symbol)

    if not raw_bids or not raw_asks:
        return result

    bids: Dict[float, float] = {float(p): float(q) for p, q in raw_bids}
    asks: Dict[float, float] = {float(p): float(q) for p, q in raw_asks}

    # Record snapshot for spoof detection
    _book_history.record(symbol, bids, asks)

    # Sorted levels
    bid_levels = sorted(bids.items(), reverse=True)[:ORDER_BOOK_DEPTH]
    ask_levels = sorted(asks.items())[:ORDER_BOOK_DEPTH]

    if not bid_levels or not ask_levels:
        return result

    best_bid = bid_levels[0][0]
    best_ask = ask_levels[0][0]

    result.mid_price    = (best_bid + best_ask) / 2
    result.spread       = best_ask - best_bid
    result.spread_pct   = result.spread / (result.mid_price + 1e-10) * 100
    result.bid_liquidity = sum(q for _, q in bid_levels)
    result.ask_liquidity = sum(q for _, q in ask_levels)
    result.bid_ask_ratio = result.bid_liquidity / (result.ask_liquidity + 1e-10)

    # Bias
    if result.bid_ask_ratio > 1.5:
        result.bias = "BID_HEAVY"
        result.confirmations.append(
            f"Order book bid-heavy: {result.bid_ask_ratio:.2f}× bid/ask ratio"
        )
    elif result.bid_ask_ratio < 0.67:
        result.bias = "ASK_HEAVY"
        result.confirmations.append(
            f"Order book ask-heavy: {result.bid_ask_ratio:.2f}× bid/ask ratio"
        )

    # Walls
    bid_walls = _detect_walls(bid_levels, "BID")
    ask_walls = _detect_walls(ask_levels, "ASK")
    result.walls = bid_walls + ask_walls

    for w in result.walls[:3]:
        result.confirmations.append(
            f"{w.side} wall @ {w.price:.4f} — {w.ratio:.1f}× avg qty"
        )

    # Clusters
    bid_clusters = _detect_clusters(bid_levels, "BID")
    ask_clusters = _detect_clusters(ask_levels, "ASK")
    result.clusters = bid_clusters + ask_clusters

    for c in result.clusters[:2]:
        result.confirmations.append(
            f"{c.side} liquidity cluster {c.price_lo:.4f}–{c.price_hi:.4f} "
            f"({c.total_qty:.2f} qty, {c.num_levels} levels)"
        )

    # Spoof detection
    spoof_alerts = _detect_spoofing(symbol, bids, asks)
    result.spoof_alerts = spoof_alerts
    for s in spoof_alerts[:2]:
        result.confirmations.append(
            f"Possible {s.side} spoof @ {s.price:.4f}: "
            f"{s.pull_pct*100:.0f}% pulled ({s.prev_qty:.2f} → {s.curr_qty:.2f})"
        )
        logger.warning(
            "Spoof alert %s %s @ %.4f: %.0f%% pulled",
            symbol, s.side, s.price, s.pull_pct * 100
        )

    # Score component (used by scoring engine)
    score = 0.0
    if result.bias == "BID_HEAVY":    score += 6
    elif result.bias == "ASK_HEAVY":  score += 6
    if bid_walls:                     score += 4   # support wall below
    if ask_walls:                     score += 4   # resistance wall above
    if bid_clusters:                  score += 3
    if spoof_alerts:                  score += 2   # directional info
    result.score_component = min(score, 15)

    return result


# ---------------------------------------------------------------------------
# Helpers for scoring / alerts
# ---------------------------------------------------------------------------

def nearest_wall_below(result: OrderBookResult, price: float) -> Optional[OrderWall]:
    """Largest bid wall below current price → potential support."""
    below = [w for w in result.walls if w.side == "BID" and w.price < price]
    return max(below, key=lambda w: w.qty) if below else None


def nearest_wall_above(result: OrderBookResult, price: float) -> Optional[OrderWall]:
    """Largest ask wall above current price → potential resistance."""
    above = [w for w in result.walls if w.side == "ASK" and w.price > price]
    return min(above, key=lambda w: w.price) if above else None
