"""
WebSocket module — real-time Binance Futures streams.
Handles aggTrade (order-flow), bookTicker (best bid/ask) and depth (order book).
Streams are restarted automatically on disconnect.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from .config import ORDER_BOOK_DEPTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_BASE = "wss://fstream.binance.com/stream"
RECONNECT_DELAY = 5          # seconds between reconnect attempts
KEEPALIVE_INTERVAL = 20      # seconds between pings
MAX_TRADE_BUFFER = 5_000     # trades kept per symbol
MAX_DEPTH_LEVELS = 20        # order book levels to maintain


# ---------------------------------------------------------------------------
# Data stores (shared across the scanner)
# ---------------------------------------------------------------------------

class StreamStore:
    """
    Thread-safe (asyncio) in-memory store for live stream data.
    Access via module-level singleton `store`.
    """

    def __init__(self) -> None:
        # symbol → deque of recent trades
        self.trades: Dict[str, Deque[Dict]] = defaultdict(
            lambda: deque(maxlen=MAX_TRADE_BUFFER)
        )
        # symbol → {"bids": {price: qty}, "asks": {price: qty}}
        self.order_books: Dict[str, Dict[str, Dict[float, float]]] = defaultdict(
            lambda: {"bids": {}, "asks": {}}
        )
        # symbol → last bookTicker
        self.book_tickers: Dict[str, Dict[str, float]] = {}

        # Callbacks registered by consumers
        self._trade_callbacks: List[Callable] = []
        self._book_callbacks:  List[Callable] = []

    def on_trade(self, cb: Callable) -> None:
        self._trade_callbacks.append(cb)

    def on_book(self, cb: Callable) -> None:
        self._book_callbacks.append(cb)

    async def _fire_trade(self, event: Dict) -> None:
        for cb in self._trade_callbacks:
            try:
                await cb(event)
            except Exception as exc:
                logger.error("Trade callback error: %s", exc)

    async def _fire_book(self, symbol: str) -> None:
        for cb in self._book_callbacks:
            try:
                await cb(symbol, self.order_books[symbol])
            except Exception as exc:
                logger.error("Book callback error: %s", exc)

    # ------------------------------------------------------------------
    # Handlers called by the WebSocket dispatcher
    # ------------------------------------------------------------------

    async def handle_agg_trade(self, msg: Dict) -> None:
        """Process an aggTrade event."""
        symbol = msg["s"].replace("USDT", "/USDT:USDT")   # normalise
        trade = {
            "ts":     msg["T"],                # trade time ms
            "price":  float(msg["p"]),
            "qty":    float(msg["q"]),
            "side":   "sell" if msg["m"] else "buy",  # m=True → maker=sell
        }
        self.trades[symbol].append(trade)
        await self._fire_trade({"symbol": symbol, **trade})

    async def handle_depth_update(self, msg: Dict) -> None:
        """Apply a diff depth update to the local order book snapshot."""
        symbol = msg["s"].replace("USDT", "/USDT:USDT")
        book = self.order_books[symbol]

        for side, key in [("b", "bids"), ("a", "asks")]:
            for price_str, qty_str in msg.get(side, []):
                price = float(price_str)
                qty   = float(qty_str)
                if qty == 0:
                    book[key].pop(price, None)
                else:
                    book[key][price] = qty

        await self._fire_book(symbol)

    async def handle_book_ticker(self, msg: Dict) -> None:
        symbol = msg["s"].replace("USDT", "/USDT:USDT")
        self.book_tickers[symbol] = {
            "bid": float(msg["b"]),
            "bid_qty": float(msg["B"]),
            "ask": float(msg["a"]),
            "ask_qty": float(msg["A"]),
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_recent_trades(self, symbol: str, n: int = 500) -> List[Dict]:
        buf = self.trades.get(symbol)
        if not buf:
            return []
        trades = list(buf)
        return trades[-n:]

    def get_order_book_snapshot(
        self, symbol: str, depth: int = MAX_DEPTH_LEVELS
    ) -> Dict[str, List]:
        book = self.order_books.get(symbol, {"bids": {}, "asks": {}})
        bids = sorted(book["bids"].items(), reverse=True)[:depth]
        asks = sorted(book["asks"].items())[:depth]
        return {
            "bids": [[p, q] for p, q in bids],
            "asks": [[p, q] for p, q in asks],
        }


store = StreamStore()


# ---------------------------------------------------------------------------
# Combined stream client
# ---------------------------------------------------------------------------

class BinanceFuturesWS:
    """
    Manages a single combined WebSocket connection to Binance Futures streams.
    Subscribes to aggTrade + depth for every active symbol.
    """

    def __init__(
        self,
        symbols: List[str],
        store: StreamStore = store,
    ) -> None:
        self._symbols = symbols
        self._store = store
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_forever())
        logger.info("WebSocket client started for %d symbols", len(self._symbols))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket client stopped")

    def update_symbols(self, symbols: List[str]) -> None:
        """Hot-swap the symbol list (takes effect on next reconnect)."""
        self._symbols = symbols

    # ------------------------------------------------------------------
    # Internal stream management
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        streams: List[str] = []
        for sym in self._symbols:
            raw = sym.replace("/USDT:USDT", "USDT").lower()  # ETHUSDT → ethusdt
            streams.append(f"{raw}@aggTrade")
            streams.append(f"{raw}@depth@100ms")
        combined = "/".join(streams)
        return f"{WS_BASE}?streams={combined}"

    async def _run_forever(self) -> None:
        while self._running:
            url = self._build_url()
            logger.info("Connecting to Binance Futures WS (%d streams)...", len(self._symbols) * 2)
            try:
                async with websockets.connect(
                    url,
                    ping_interval=KEEPALIVE_INTERVAL,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("WebSocket connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._dispatch(raw)
            except (ConnectionClosedError, ConnectionClosedOK) as exc:
                logger.warning("WS disconnected (%s). Reconnecting in %ds…", exc, RECONNECT_DELAY)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("WS unexpected error: %s — reconnecting in %ds", exc, RECONNECT_DELAY)

            if self._running:
                await asyncio.sleep(RECONNECT_DELAY)

    async def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
            data = msg.get("data", msg)
            stream_name = msg.get("stream", "")

            if "aggTrade" in stream_name:
                await self._store.handle_agg_trade(data)
            elif "depth" in stream_name:
                await self._store.handle_depth_update(data)
            elif "bookTicker" in stream_name:
                await self._store.handle_book_ticker(data)
        except Exception as exc:
            logger.debug("WS dispatch error: %s", exc)


# ---------------------------------------------------------------------------
# Kline (candlestick) stream — separate lightweight connection
# ---------------------------------------------------------------------------

class KlineStream:
    """
    Subscribes to kline streams so indicators can be updated in real time.
    Fires the `on_candle_close` callback with a completed candle dict.
    """

    def __init__(self, symbols: List[str], timeframe: str = "1m") -> None:
        self._symbols = symbols
        self._timeframe = timeframe
        self._callbacks: List[Callable] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def on_candle_close(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _build_url(self) -> str:
        tf_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "1h": "1h", "4h": "4h", "1d": "1d",
        }
        tf = tf_map.get(self._timeframe, "1m")
        streams = [
            f"{sym.replace('/USDT:USDT','USDT').lower()}@kline_{tf}"
            for sym in self._symbols
        ]
        return f"{WS_BASE}?streams={'/'.join(streams)}"

    async def _run_forever(self) -> None:
        while self._running:
            try:
                async with websockets.connect(
                    self._build_url(),
                    ping_interval=KEEPALIVE_INTERVAL,
                ) as ws:
                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        k = data.get("k", {})
                        if k.get("x"):  # candle closed
                            candle = {
                                "symbol":    data["s"].replace("USDT", "/USDT:USDT"),
                                "timeframe": self._timeframe,
                                "ts":        k["t"],
                                "open":      float(k["o"]),
                                "high":      float(k["h"]),
                                "low":       float(k["l"]),
                                "close":     float(k["c"]),
                                "volume":    float(k["v"]),
                            }
                            for cb in self._callbacks:
                                try:
                                    await cb(candle)
                                except Exception as exc:
                                    logger.error("Kline callback error: %s", exc)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("KlineStream error: %s", exc)
            if self._running:
                await asyncio.sleep(RECONNECT_DELAY)
