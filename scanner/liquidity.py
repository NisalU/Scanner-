"""
Liquidity & Order Book Analysis — Professional Grade
─────────────────────────────────────────────────────
• Wall and cluster detection (unchanged public API)
• Multi-factor spoof probability scoring — only high-confidence events
  (score ≥ MIN_SPOOF_SCORE) are surfaced; normal cancellations are ignored

Classes
-------
OrderTracker   — tracks order lifetimes (appear → vanish → lifetime)
SpoofDetector  — 6-factor scoring engine, one singleton per process
LiquidityEvent — rich event model replacing the old SpoofAlert

Scoring factors (100 pts total)
--------------------------------
A) Size anomaly      25 pts  order > N× avg depth
B) Lifetime          25 pts  0.5 s – 5 s window scores highest
C) Distance from mid 15 pts  ≥ MIN_DISTANCE_PERCENT from best price
D) Cancel rate       15 pts  cancelled_vol / total_vol for that level
E) Repeat pattern    10 pts  same zone appearing + vanishing 3+ times
F) Price reaction    10 pts  price moved adversely after removal
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    ORDER_BOOK_DEPTH,
    WALL_THRESHOLD_MULT,
    MIN_ORDER_SIZE_MULTIPLIER,
    MIN_LIFETIME_SECONDS,
    MAX_LIFETIME_SECONDS,
    MIN_DISTANCE_PERCENT,
    MIN_REPEAT_COUNT,
    MIN_SPOOF_SCORE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class OrderWall:
    side:  str
    price: float
    qty:   float
    ratio: float   # qty / avg_qty at that side


@dataclass
class LiquidityCluster:
    side:       str
    price_lo:   float
    price_hi:   float
    total_qty:  float
    num_levels: int


@dataclass
class LiquidityEvent:
    """
    High-confidence liquidity manipulation event.
    Only created when spoof_score >= MIN_SPOOF_SCORE.
    """
    symbol:         str
    side:           str        # 'BID' | 'ASK'
    price:          float
    order_size_usd: float      # notional value at mid price
    lifetime_s:     float      # seconds the order was visible
    cancel_rate:    float      # cancelled_vol / (cancelled + executed)
    repeats:        int        # times this pattern repeated at same zone
    price_impact:   float      # % price change after removal
    spoof_score:    int        # 0–100

    # ── Backward-compat properties (old SpoofAlert fields) ──────────────────
    @property
    def prev_qty(self) -> float:        # original qty before removal
        return self.order_size_usd

    @property
    def curr_qty(self) -> float:        # qty after removal (always 0)
        return 0.0

    @property
    def pull_pct(self) -> float:        # alias for cancel_rate
        return self.cancel_rate


# Keep old name as alias so nothing outside liquidity.py breaks
SpoofAlert = LiquidityEvent


@dataclass
class OrderBookResult:
    symbol:          str
    bid_liquidity:   float = 0.0
    ask_liquidity:   float = 0.0
    bid_ask_ratio:   float = 0.0
    spread:          float = 0.0
    spread_pct:      float = 0.0
    mid_price:       float = 0.0
    walls:           List[OrderWall]        = field(default_factory=list)
    clusters:        List[LiquidityCluster] = field(default_factory=list)
    spoof_alerts:    List[LiquidityEvent]   = field(default_factory=list)
    bias:            str = "NEUTRAL"        # BID_HEAVY | ASK_HEAVY | NEUTRAL
    confirmations:   List[str]              = field(default_factory=list)
    score_component: float = 0.0


# ---------------------------------------------------------------------------
# OrderTracker — per-order lifetime accounting
# ---------------------------------------------------------------------------

@dataclass
class _TrackedOrder:
    price:      float
    qty:        float
    first_seen: float   # monotonic timestamp
    last_seen:  float


class OrderTracker:
    """
    Diffs consecutive order-book snapshots to determine when orders
    appeared and disappeared, computing their visible lifetime.

    Usage
    -----
    disappeared = tracker.update(symbol, "BID", bids_dict, now)
    # disappeared: [(price, qty, lifetime_s), ...]
    """

    def __init__(self, max_expired: int = 500) -> None:
        # symbol → side → price → TrackedOrder
        self._live: Dict[str, Dict[str, Dict[float, _TrackedOrder]]] = defaultdict(
            lambda: {"BID": {}, "ASK": {}}
        )
        # rolling history of expired orders
        self._expired: Dict[str, Dict[str, Deque[Tuple[float, float, float]]]] = (
            defaultdict(lambda: {
                "BID": deque(maxlen=max_expired),
                "ASK": deque(maxlen=max_expired),
            })
        )

    def update(
        self,
        symbol: str,
        side: str,
        levels: Dict[float, float],
        now: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Returns list of (price, qty, lifetime_s) for orders that just vanished.
        Also updates last_seen for orders still present.
        """
        live = self._live[symbol][side]
        disappeared: List[Tuple[float, float, float]] = []

        for price, qty in levels.items():
            if price in live:
                live[price].last_seen = now
                live[price].qty = qty
            else:
                live[price] = _TrackedOrder(price, qty, now, now)

        for price in list(live):
            if price not in levels:
                order = live.pop(price)
                lifetime = order.last_seen - order.first_seen
                disappeared.append((price, order.qty, lifetime))
                self._expired[symbol][side].append((price, order.qty, lifetime))

        return disappeared

    def get_expired(
        self, symbol: str, side: str
    ) -> Deque[Tuple[float, float, float]]:
        return self._expired[symbol][side]


# ---------------------------------------------------------------------------
# SpoofDetector — 6-factor probability scoring
# ---------------------------------------------------------------------------

class SpoofDetector:
    """
    Evaluates disappeared orders against 6 factors and emits
    LiquidityEvent only when the composite score >= MIN_SPOOF_SCORE.

    All state is per-symbol so symbols never interfere with each other.
    """

    def __init__(self) -> None:
        self._tracker = OrderTracker()

        # Repeat counter: symbol → side → price_zone → count
        self._repeats: Dict[str, Dict[str, Dict[float, int]]] = defaultdict(
            lambda: {"BID": defaultdict(int), "ASK": defaultdict(int)}
        )
        # Price history for reaction check: symbol → deque[(mid, monotonic_ts)]
        self._prices: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=60)
        )
        # Volume accounting: symbol → side → price → [cancelled, executed]
        self._vol: Dict[str, Dict[str, Dict[float, List[float]]]] = defaultdict(
            lambda: {"BID": defaultdict(lambda: [0.0, 0.0]),
                     "ASK": defaultdict(lambda: [0.0, 0.0])}
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _zone(price: float, mid: float) -> float:
        """Snap price to 0.1 % grid for repeat detection."""
        tick = mid * 0.001
        return round(price / tick) * tick if tick > 0 else price

    # ── individual factor scorers ────────────────────────────────────────────

    def _factor_size(self, qty: float, avg: float) -> float:
        """A — size anomaly: 0–25 pts."""
        if avg <= 0 or qty < avg * MIN_ORDER_SIZE_MULTIPLIER:
            return 0.0
        ratio = qty / avg
        return min(25.0, 5.0 + (ratio / MIN_ORDER_SIZE_MULTIPLIER) * 20.0)

    @staticmethod
    def _factor_lifetime(lifetime_s: float) -> float:
        """B — short-lived order: 0–25 pts."""
        if lifetime_s < MIN_LIFETIME_SECONDS or lifetime_s > MAX_LIFETIME_SECONDS:
            return 0.0
        if lifetime_s <= 1.0:  return 25.0
        if lifetime_s <= 2.0:  return 20.0
        if lifetime_s <= 3.0:  return 15.0
        return 10.0

    @staticmethod
    def _factor_distance(price: float, mid: float) -> float:
        """C — distance from mid: 0–15 pts."""
        if mid <= 0:
            return 0.0
        dist = abs(price - mid) / mid * 100
        if dist < MIN_DISTANCE_PERCENT:  return 0.0
        if dist >= 1.0:                  return 15.0
        if dist >= 0.5:                  return 10.0
        return 5.0

    def _factor_cancel_rate(
        self, symbol: str, side: str, price: float, qty: float
    ) -> float:
        """D — cancel-to-trade ratio at this level: 0–15 pts."""
        bucket = self._vol[symbol][side][price]
        bucket[0] += qty          # cancelled volume
        total = bucket[0] + bucket[1]
        if total <= 0:
            return 15.0           # never traded → worst case
        rate = bucket[0] / total
        if rate >= 0.95:  return 15.0
        if rate >= 0.80:  return 10.0
        if rate >= 0.60:  return  5.0
        return 0.0

    def _factor_repeats(
        self, symbol: str, side: str, price: float, mid: float
    ) -> Tuple[float, int]:
        """E — repeated pattern at same zone: 0–10 pts. Returns (score, count)."""
        zone = self._zone(price, mid)
        self._repeats[symbol][side][zone] += 1
        n = self._repeats[symbol][side][zone]
        if n >= MIN_REPEAT_COUNT:  return 10.0, n
        if n == 2:                 return  5.0, n
        return 0.0, n

    def _factor_price_reaction(
        self, symbol: str, side: str, now: float
    ) -> Tuple[float, float]:
        """F — adverse price move after removal: 0–10 pts. Returns (score, impact_pct)."""
        hist = self._prices[symbol]
        if len(hist) < 2:
            return 0.0, 0.0

        recent = hist[-1][0]
        # Price ~3 s before the cancellation
        ref = next((p for p, t in reversed(hist) if now - t >= 3.0), hist[0][0])
        if ref <= 0:
            return 0.0, 0.0

        impact = (recent - ref) / ref * 100

        if side == "BID" and impact < -0.10:   # bid pulled → price fell
            return 10.0, impact
        if side == "ASK" and impact >  0.10:   # ask pulled → price rose
            return 10.0, impact
        return 0.0, impact

    # ── main entry point ─────────────────────────────────────────────────────

    def process_snapshot(
        self,
        symbol:  str,
        bids:    Dict[float, float],
        asks:    Dict[float, float],
        mid:     float,
        now:     Optional[float] = None,
    ) -> List[LiquidityEvent]:
        if now is None:
            now = time.monotonic()

        self._prices[symbol].append((mid, now))

        avg_bid = float(np.mean(list(bids.values()))) if bids else 1.0
        avg_ask = float(np.mean(list(asks.values()))) if asks else 1.0

        events: List[LiquidityEvent] = []

        for side, levels, avg_qty in (
            ("BID", bids, avg_bid),
            ("ASK", asks, avg_ask),
        ):
            for price, qty, lifetime_s in self._tracker.update(
                symbol, side, levels, now
            ):
                # Fast-path: skip orders that can't score high enough
                if qty < avg_qty * MIN_ORDER_SIZE_MULTIPLIER:
                    continue
                if not (MIN_LIFETIME_SECONDS <= lifetime_s <= MAX_LIFETIME_SECONDS):
                    continue

                # Score all factors
                sa = self._factor_size(qty, avg_qty)
                sb = self._factor_lifetime(lifetime_s)
                sc = self._factor_distance(price, mid)
                sd = self._factor_cancel_rate(symbol, side, price, qty)
                se, reps = self._factor_repeats(symbol, side, price, mid)
                sf, impact = self._factor_price_reaction(symbol, side, now)

                total = int(sa + sb + sc + sd + se + sf)

                if total < MIN_SPOOF_SCORE:
                    continue

                bucket = self._vol[symbol][side][price]
                total_vol = bucket[0] + bucket[1]
                cancel_rate = bucket[0] / total_vol if total_vol > 0 else 1.0

                event = LiquidityEvent(
                    symbol=symbol,
                    side=side,
                    price=price,
                    order_size_usd=round(qty * mid, 2),
                    lifetime_s=round(lifetime_s, 2),
                    cancel_rate=round(cancel_rate, 4),
                    repeats=reps,
                    price_impact=round(impact, 4),
                    spoof_score=total,
                )
                events.append(event)
                _log_event(event)

        return events


# Module-level singleton — shared across all analyse_order_book() calls
_detector = SpoofDetector()


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def _log_event(e: LiquidityEvent) -> None:
    pair = (
        e.symbol
        .replace("/USDT:USDT", "USDT")
        .replace("/USDT", "USDT")
    )
    if e.order_size_usd >= 1_000_000:
        size_str = f"${e.order_size_usd / 1_000_000:.2f}M"
    else:
        size_str = f"${e.order_size_usd / 1_000:.0f}K"

    impact_str = f"{e.price_impact:+.2f}%" if e.price_impact != 0 else "n/a"

    logger.warning(
        "\n"
        "  ── Liquidity Manipulation Detected ──────────────────────\n"
        "  PAIR:         %s\n"
        "  SIDE:         %s\n"
        "  PRICE:        %g\n"
        "  ORDER SIZE:   %s\n"
        "  LIFETIME:     %.2f s\n"
        "  CANCEL RATE:  %.0f%%\n"
        "  REPEATS:      %d\n"
        "  PRICE IMPACT: %s\n"
        "  SPOOF SCORE:  %d/100\n"
        "  ─────────────────────────────────────────────────────────",
        pair, e.side, e.price, size_str,
        e.lifetime_s, e.cancel_rate * 100,
        e.repeats, impact_str, e.spoof_score,
    )


# ---------------------------------------------------------------------------
# Wall and cluster detection (public API unchanged)
# ---------------------------------------------------------------------------

def _detect_walls(
    levels: List[Tuple[float, float]],
    side: str,
    threshold_mult: float = WALL_THRESHOLD_MULT,
) -> List[OrderWall]:
    if not levels:
        return []
    qtys = [q for _, q in levels]
    avg  = float(np.mean(qtys)) + 1e-10
    walls = [
        OrderWall(side=side, price=p, qty=q, ratio=q / avg)
        for p, q in levels
        if q >= avg * threshold_mult
    ]
    return sorted(walls, key=lambda w: w.ratio, reverse=True)


def _detect_clusters(
    levels: List[Tuple[float, float]],
    side: str,
    cluster_pct: float = 0.002,
) -> List[LiquidityCluster]:
    if not levels:
        return []
    clusters: List[LiquidityCluster] = []
    lo = hi = levels[0][0]
    total = levels[0][1]
    count = 1
    for price, qty in levels[1:]:
        if abs(price - hi) / (hi + 1e-10) <= cluster_pct:
            total += qty
            hi     = max(hi, price)
            lo     = min(lo, price)
            count += 1
        else:
            if count >= 3:
                clusters.append(LiquidityCluster(side, lo, hi, total, count))
            lo = hi = price
            total = qty
            count = 1
    if count >= 3:
        clusters.append(LiquidityCluster(side, lo, hi, total, count))
    return sorted(clusters, key=lambda c: c.total_qty, reverse=True)


# ---------------------------------------------------------------------------
# Public entry point (API unchanged)
# ---------------------------------------------------------------------------

def analyse_order_book(
    symbol:   str,
    raw_bids: List[List[float]],
    raw_asks: List[List[float]],
) -> OrderBookResult:
    """
    Full order-book analysis.
    raw_bids / raw_asks are [[price, qty], ...] as returned by ccxt.
    """
    result = OrderBookResult(symbol=symbol)

    if not raw_bids or not raw_asks:
        return result

    bids: Dict[float, float] = {float(p): float(q) for p, q in raw_bids}
    asks: Dict[float, float] = {float(p): float(q) for p, q in raw_asks}

    bid_levels = sorted(bids.items(), reverse=True)[:ORDER_BOOK_DEPTH]
    ask_levels = sorted(asks.items())[:ORDER_BOOK_DEPTH]

    if not bid_levels or not ask_levels:
        return result

    best_bid = bid_levels[0][0]
    best_ask = ask_levels[0][0]

    result.mid_price     = (best_bid + best_ask) / 2
    result.spread        = best_ask - best_bid
    result.spread_pct    = result.spread / (result.mid_price + 1e-10) * 100
    result.bid_liquidity = sum(q for _, q in bid_levels)
    result.ask_liquidity = sum(q for _, q in ask_levels)
    result.bid_ask_ratio = result.bid_liquidity / (result.ask_liquidity + 1e-10)

    # ── Bias ────────────────────────────────────────────────────────────────
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

    # ── Walls ────────────────────────────────────────────────────────────────
    result.walls = (
        _detect_walls(bid_levels, "BID") +
        _detect_walls(ask_levels, "ASK")
    )
    for w in result.walls[:3]:
        result.confirmations.append(
            f"{w.side} wall @ {w.price:.4g} — {w.ratio:.1f}× avg qty"
        )

    # ── Clusters ─────────────────────────────────────────────────────────────
    result.clusters = (
        _detect_clusters(bid_levels, "BID") +
        _detect_clusters(ask_levels, "ASK")
    )
    for c in result.clusters[:2]:
        result.confirmations.append(
            f"{c.side} cluster {c.price_lo:.4g}–{c.price_hi:.4g} "
            f"({c.total_qty:.2f} qty, {c.num_levels} levels)"
        )

    # ── Multi-factor spoof detection ─────────────────────────────────────────
    events = _detector.process_snapshot(
        symbol, bids, asks, result.mid_price
    )
    result.spoof_alerts = events
    for e in events[:2]:
        result.confirmations.append(
            f"Liquidity manipulation {e.side} @ {e.price:.4g} — "
            f"score {e.spoof_score}/100  lifetime {e.lifetime_s:.1f}s  "
            f"repeats {e.repeats}"
        )

    # ── Score component ───────────────────────────────────────────────────────
    s = 0.0
    if result.bias in ("BID_HEAVY", "ASK_HEAVY"):   s += 6
    if any(w.side == "BID" for w in result.walls):  s += 4
    if any(w.side == "ASK" for w in result.walls):  s += 4
    if result.clusters:                             s += 3
    if events:                                      s += 3   # high-confidence only
    result.score_component = min(s, 15)

    return result


# ---------------------------------------------------------------------------
# Helpers for the scoring engine (public API unchanged)
# ---------------------------------------------------------------------------

def nearest_wall_below(result: OrderBookResult, price: float) -> Optional[OrderWall]:
    """Largest bid wall below current price → potential support."""
    below = [w for w in result.walls if w.side == "BID" and w.price < price]
    return max(below, key=lambda w: w.qty) if below else None


def nearest_wall_above(result: OrderBookResult, price: float) -> Optional[OrderWall]:
    """Largest ask wall above current price → potential resistance."""
    above = [w for w in result.walls if w.side == "ASK" and w.price > price]
    return min(above, key=lambda w: w.price) if above else None
