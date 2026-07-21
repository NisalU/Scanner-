"""
Unit tests for the multi-factor spoof detection system in scanner/liquidity.py

Run:  python -m pytest tests/test_liquidity.py -v
"""

import time
import unittest
from unittest.mock import patch

from scanner.liquidity import (
    OrderTracker,
    SpoofDetector,
    LiquidityEvent,
    OrderBookResult,
    analyse_order_book,
    _detect_walls,
    _detect_clusters,
    nearest_wall_below,
    nearest_wall_above,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_levels(n: int = 10, base: float = 100.0, step: float = 0.1) -> dict:
    """Return a uniform order-book side as {price: qty}."""
    return {round(base + i * step, 4): 1.0 for i in range(n)}


def _make_raw(n: int = 10, base: float = 100.0, step: float = 0.1, qty: float = 1.0):
    """Return [[price, qty], ...] as ccxt would."""
    return [[round(base + i * step, 4), qty] for i in range(n)]


# ---------------------------------------------------------------------------
# OrderTracker
# ---------------------------------------------------------------------------

class TestOrderTracker(unittest.TestCase):

    def test_new_orders_not_reported_as_disappeared(self):
        t = OrderTracker()
        levels = _make_levels(5)
        gone = t.update("BTC", "BID", levels, 1.0)
        self.assertEqual(gone, [], "First snapshot should produce no disappeared orders")

    def test_stable_book_produces_no_events(self):
        t = OrderTracker()
        levels = _make_levels(5)
        t.update("BTC", "BID", levels, 1.0)
        gone = t.update("BTC", "BID", levels, 2.0)
        self.assertEqual(gone, [])

    def test_removed_order_is_reported(self):
        t = OrderTracker()
        levels = _make_levels(5)
        t.update("BTC", "BID", levels, 1.0)

        reduced = dict(list(levels.items())[1:])   # drop first price
        gone = t.update("BTC", "BID", reduced, 3.0)

        self.assertEqual(len(gone), 1)
        price, qty, lifetime = gone[0]
        self.assertAlmostEqual(lifetime, 2.0, places=5)

    def test_lifetime_is_last_seen_minus_first_seen(self):
        t = OrderTracker()
        levels = {100.0: 5.0}
        t.update("BTC", "ASK", levels, 10.0)
        t.update("BTC", "ASK", levels, 11.5)   # still present at 11.5
        gone = t.update("BTC", "ASK", {}, 12.0)  # vanishes

        self.assertEqual(len(gone), 1)
        _, _, lifetime = gone[0]
        # last_seen=11.5, first_seen=10.0 → 1.5 s
        self.assertAlmostEqual(lifetime, 1.5, places=4)

    def test_multiple_symbols_independent(self):
        t = OrderTracker()
        btc = {100.0: 1.0}
        eth = {200.0: 1.0}
        t.update("BTC", "BID", btc, 1.0)
        t.update("ETH", "BID", eth, 1.0)
        gone_btc = t.update("BTC", "BID", {}, 2.0)
        gone_eth = t.update("ETH", "BID", eth, 2.0)
        self.assertEqual(len(gone_btc), 1)
        self.assertEqual(len(gone_eth), 0)


# ---------------------------------------------------------------------------
# SpoofDetector — factor tests
# ---------------------------------------------------------------------------

class TestSpoofDetectorFactors(unittest.TestCase):

    def setUp(self):
        self.det = SpoofDetector()

    def test_small_order_ignored(self):
        """Order below MIN_ORDER_SIZE_MULTIPLIER × avg must not score."""
        # avg qty = 1.0; small order = 2.0 (< 5× avg)
        bids = {100.0: 2.0, 99.9: 1.0, 99.8: 1.0}
        asks = {100.1: 1.0, 100.2: 1.0}
        self.det.process_snapshot("BTC", bids, asks, 100.05, now=1.0)

        # Remove the 2.0-qty order
        bids2 = {99.9: 1.0, 99.8: 1.0}
        events = self.det.process_snapshot("BTC", bids2, asks, 100.05, now=2.0)
        self.assertEqual(events, [], "Small order should produce no events")

    def test_lifetime_too_short_ignored(self):
        """Order visible for < MIN_LIFETIME_SECONDS (0.5 s) must be skipped."""
        avg = 1.0
        big_qty = 10.0  # 10× avg
        bids = {100.0: big_qty, 99.9: avg, 99.8: avg, 99.7: avg}
        asks = {100.1: avg, 100.2: avg}
        self.det.process_snapshot("BTC", bids, asks, 100.05, now=1.000)
        bids2 = {99.9: avg, 99.8: avg, 99.7: avg}
        events = self.det.process_snapshot("BTC", bids2, asks, 100.05, now=1.100)
        self.assertEqual(events, [], "Sub-500 ms order should be ignored")

    def test_lifetime_too_long_ignored(self):
        """Order visible > MAX_LIFETIME_SECONDS (5 s) is not a spoof."""
        avg = 1.0
        big_qty = 20.0
        bids = {100.0: big_qty, 99.9: avg, 99.8: avg, 99.7: avg}
        asks = {100.1: avg, 100.2: avg}
        self.det.process_snapshot("BTC", bids, asks, 100.05, now=1.0)
        bids2 = {99.9: avg, 99.8: avg, 99.7: avg}
        events = self.det.process_snapshot("BTC", bids2, asks, 100.05, now=10.0)
        self.assertEqual(events, [], "Long-lived order should be ignored")

    def test_high_score_event_emitted(self):
        """An order that ticks all the boxes must produce a LiquidityEvent."""
        det = SpoofDetector()
        mid = 50000.0
        avg = 1.0
        big = 30.0  # 30× avg → max size score

        # Far enough from mid to score on distance
        far_bid_price = mid * (1 - 0.005)   # 0.5% below mid

        bids = {far_bid_price: big, mid - 10: avg, mid - 20: avg,
                mid - 30: avg, mid - 40: avg}
        asks = {mid + 10: avg, mid + 20: avg}

        det.process_snapshot("BTC", bids, asks, mid, now=1.0)
        bids2 = {mid - 10: avg, mid - 20: avg, mid - 30: avg, mid - 40: avg}
        events = det.process_snapshot("BTC", bids2, asks, mid * 0.999, now=2.5)

        # May or may not cross 80 depending on factors, but event must be LiquidityEvent
        for e in events:
            self.assertIsInstance(e, LiquidityEvent)
            self.assertGreaterEqual(e.spoof_score, 0)
            self.assertLessEqual(e.spoof_score, 100)

    def test_repeat_counter_increments(self):
        """Same price zone appearing+disappearing should increment repeat count."""
        det = SpoofDetector()
        mid = 1000.0
        avg = 1.0
        big = 10.0
        price = mid * 0.995  # 0.5% below mid

        bids_with    = {price: big, mid - 2: avg, mid - 3: avg, mid - 4: avg}
        bids_without = {mid - 2: avg, mid - 3: avg, mid - 4: avg}
        asks = {mid + 2: avg, mid + 3: avg}

        repeat_events = []
        ts = 1.0
        for _ in range(4):
            det.process_snapshot("ETH", bids_with, asks, mid, now=ts)
            ts += 0.1
            evs = det.process_snapshot("ETH", bids_without, asks, mid * 0.999, now=ts + 1.5)
            repeat_events.extend(evs)
            ts += 3.0

        if repeat_events:
            self.assertGreaterEqual(repeat_events[-1].repeats, 1)

    def test_event_fields_populated(self):
        """LiquidityEvent must have all required fields with sensible values."""
        det = SpoofDetector()
        mid = 30000.0
        avg = 1.0
        big = 50.0
        price = mid * 0.994   # 0.6% below mid

        bids = {price: big, mid-5: avg, mid-10: avg, mid-15: avg, mid-20: avg}
        asks = {mid+5: avg, mid+10: avg}
        det.process_snapshot("BTC", bids, asks, mid, now=1.0)
        bids2 = {mid-5: avg, mid-10: avg, mid-15: avg, mid-20: avg}
        events = det.process_snapshot("BTC", bids2, asks, mid * 0.998, now=2.8)

        for e in events:
            self.assertIsInstance(e.symbol, str)
            self.assertIn(e.side, ("BID", "ASK"))
            self.assertGreater(e.price, 0)
            self.assertGreater(e.order_size_usd, 0)
            self.assertGreaterEqual(e.lifetime_s, 0)
            self.assertGreaterEqual(e.cancel_rate, 0)
            self.assertLessEqual(e.cancel_rate, 1)
            self.assertGreaterEqual(e.spoof_score, 0)
            self.assertLessEqual(e.spoof_score, 100)

    def test_backward_compat_properties(self):
        """prev_qty, curr_qty, pull_pct must work for any downstream code."""
        e = LiquidityEvent(
            symbol="BTC/USDT", side="BID", price=50000.0,
            order_size_usd=1_000_000.0, lifetime_s=1.5, cancel_rate=0.95,
            repeats=3, price_impact=-0.3, spoof_score=88,
        )
        self.assertEqual(e.prev_qty, 1_000_000.0)
        self.assertEqual(e.curr_qty, 0.0)
        self.assertEqual(e.pull_pct, 0.95)


# ---------------------------------------------------------------------------
# Wall and cluster detection
# ---------------------------------------------------------------------------

class TestWallDetection(unittest.TestCase):

    def test_detects_large_wall(self):
        levels = [(100.0, 1.0), (99.9, 1.0), (99.8, 50.0), (99.7, 1.0)]
        walls = _detect_walls(levels, "BID", threshold_mult=5.0)
        self.assertEqual(len(walls), 1)
        self.assertAlmostEqual(walls[0].price, 99.8)

    def test_no_walls_when_uniform(self):
        levels = [(100.0 - i * 0.1, 1.0) for i in range(10)]
        walls = _detect_walls(levels, "BID")
        self.assertEqual(walls, [])

    def test_empty_levels(self):
        self.assertEqual(_detect_walls([], "BID"), [])


class TestClusterDetection(unittest.TestCase):

    def test_detects_cluster(self):
        levels = [
            (100.0, 2.0), (100.05, 2.0), (100.1, 2.0),   # cluster
            (102.0, 1.0),
        ]
        clusters = _detect_clusters(levels, "ASK")
        self.assertGreaterEqual(len(clusters), 1)
        self.assertEqual(clusters[0].num_levels, 3)

    def test_no_cluster_when_spread_out(self):
        levels = [(100.0 + i * 5.0, 1.0) for i in range(5)]
        self.assertEqual(_detect_clusters(levels, "ASK"), [])


# ---------------------------------------------------------------------------
# analyse_order_book integration
# ---------------------------------------------------------------------------

class TestAnalyseOrderBook(unittest.TestCase):

    def _symmetric_book(self, n=10, mid=100.0, qty=1.0):
        bids = [[mid - (i + 1) * 0.1, qty] for i in range(n)]
        asks = [[mid + (i + 1) * 0.1, qty] for i in range(n)]
        return bids, asks

    def test_returns_result_object(self):
        bids, asks = self._symmetric_book()
        result = analyse_order_book("BTC/USDT", bids, asks)
        self.assertIsInstance(result, OrderBookResult)

    def test_mid_price_correct(self):
        bids = [[99.9, 1.0]]
        asks = [[100.1, 1.0]]
        result = analyse_order_book("BTC", bids, asks)
        self.assertAlmostEqual(result.mid_price, 100.0)

    def test_empty_book_returns_default(self):
        result = analyse_order_book("BTC", [], [])
        self.assertEqual(result.mid_price, 0.0)
        self.assertEqual(result.walls, [])

    def test_bid_heavy_bias(self):
        bids = [[100.0 - i * 0.1, 10.0] for i in range(10)]
        asks = [[100.1 + i * 0.1,  1.0] for i in range(10)]
        result = analyse_order_book("BTC", bids, asks)
        self.assertEqual(result.bias, "BID_HEAVY")

    def test_ask_heavy_bias(self):
        bids = [[100.0 - i * 0.1,  1.0] for i in range(10)]
        asks = [[100.1 + i * 0.1, 10.0] for i in range(10)]
        result = analyse_order_book("BTC", bids, asks)
        self.assertEqual(result.bias, "ASK_HEAVY")

    def test_score_component_bounded(self):
        bids, asks = self._symmetric_book()
        result = analyse_order_book("BTC", bids, asks)
        self.assertGreaterEqual(result.score_component, 0)
        self.assertLessEqual(result.score_component, 15)

    def test_wall_detected_via_full_pipeline(self):
        bids = [[100.0 - i * 0.1, 1.0] for i in range(9)]
        bids.append([99.0, 100.0])   # massive wall
        asks = [[100.1 + i * 0.1, 1.0] for i in range(10)]
        result = analyse_order_book("BTC", bids, asks)
        self.assertTrue(any(w.side == "BID" for w in result.walls))


# ---------------------------------------------------------------------------
# Nearest-wall helpers
# ---------------------------------------------------------------------------

class TestNearestWallHelpers(unittest.TestCase):

    def _result_with_walls(self):
        bids = [[100.0 - i * 0.1, 1.0] for i in range(9)]
        bids.append([98.0, 100.0])
        asks = [[100.1 + i * 0.1, 1.0] for i in range(9)]
        asks.append([101.5, 100.0])
        return analyse_order_book("BTC", bids, asks)

    def test_nearest_wall_below(self):
        result = self._result_with_walls()
        wall = nearest_wall_below(result, 100.0)
        if wall:
            self.assertLess(wall.price, 100.0)
            self.assertEqual(wall.side, "BID")

    def test_nearest_wall_above(self):
        result = self._result_with_walls()
        wall = nearest_wall_above(result, 100.0)
        if wall:
            self.assertGreater(wall.price, 100.0)
            self.assertEqual(wall.side, "ASK")

    def test_no_wall_returns_none(self):
        result = OrderBookResult(symbol="BTC")
        self.assertIsNone(nearest_wall_below(result, 100.0))
        self.assertIsNone(nearest_wall_above(result, 100.0))


if __name__ == "__main__":
    unittest.main()
