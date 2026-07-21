"""
Exchange module — abstracts REST API calls through ccxt.
Designed to be exchange-agnostic; add new exchanges by subclassing BaseExchange.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import ccxt.async_support as ccxt
import pandas as pd

from .config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TESTNET,
    CANDLE_LOOKBACK,
    MIN_VOLUME_USDT,
    MAX_PAIRS,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseExchange(ABC):
    """Minimum interface every exchange adapter must implement."""

    @abstractmethod
    async def init(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def get_usdt_futures_pairs(self) -> List[str]: ...

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, timeframe: str, limit: int = CANDLE_LOOKBACK
    ) -> pd.DataFrame: ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]: ...

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]: ...

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> Optional[float]: ...


# ---------------------------------------------------------------------------
# Binance Futures adapter
# ---------------------------------------------------------------------------

class BinanceFuturesExchange(BaseExchange):
    """
    Binance USDT-M perpetual futures via ccxt async.
    Uses public endpoints only for scanning (no order placement).
    """

    def __init__(self) -> None:
        options: Dict[str, Any] = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }
        if BINANCE_TESTNET:
            options["sandboxMode"] = True

        self._ex = ccxt.binanceusdm(
            {
                "apiKey":  BINANCE_API_KEY  or None,
                "secret":  BINANCE_API_SECRET or None,
                "options": options,
                "enableRateLimit": True,
            }
        )
        self._markets_loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        await self._ex.load_markets()
        self._markets_loaded = True
        logger.info(
            "Binance USDT-M loaded — %d instruments", len(self._ex.markets)
        )

    async def close(self) -> None:
        await self._ex.close()
        logger.info("Exchange connection closed")

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    async def get_usdt_futures_pairs(self) -> List[str]:
        """Return top-volume USDT-M perpetual symbols."""
        if not self._markets_loaded:
            await self.init()

        # Step 1 — filter active USDT-M perps from market metadata (no ticker needed)
        # Use `linear=True` (USDT-margined) + `expiry is None` (perpetual, not dated).
        perps = [
            sym for sym, mkt in self._ex.markets.items()
            if (
                mkt.get("active")
                and mkt.get("linear")           # USDT-M (linear), not inverse/coin-m
                and mkt.get("expiry") is None   # perpetual only, exclude dated futures
            )
        ]
        if not perps:
            # Log sample market entries so the format can be diagnosed
            sample = list(self._ex.markets.items())[:3]
            for s, m in sample:
                logger.warning(
                    "Market sample — symbol=%s type=%s linear=%s settle=%s "
                    "active=%s expiry=%s",
                    s, m.get("type"), m.get("linear"), m.get("settle"),
                    m.get("active"), m.get("expiry"),
                )
            logger.warning("No active USDT-M perpetuals found in markets")
            return []

        # Step 2 — fetch tickers to rank by volume
        tickers: Dict[str, Any] = await self._ex.fetch_tickers()

        candidates: List[Tuple[str, float]] = []
        for symbol in perps:
            ticker = tickers.get(symbol, {})
            info   = ticker.get("info", {})

            # Try normalized fields first, then raw Binance info dict
            vol24: float = (
                ticker.get("quoteVolume")
                or float(info.get("quoteVolume") or 0)
                or (ticker.get("baseVolume") or 0) * (ticker.get("last") or 0)
                or float(info.get("volume") or 0) * float(info.get("lastPrice") or 0)
            )
            if vol24 >= MIN_VOLUME_USDT:
                candidates.append((symbol, vol24))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            pairs = [sym for sym, _ in candidates[:MAX_PAIRS]]
        else:
            # Volume data unavailable — take all active perps up to MAX_PAIRS
            logger.warning(
                "Volume data unavailable from tickers; "
                "falling back to all %d active USDT-M perps (capped at %d).",
                len(perps), MAX_PAIRS,
            )
            pairs = sorted(perps)[:MAX_PAIRS]

        logger.info("Universe: %d USDT-M pairs selected", len(pairs))
        return pairs

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = CANDLE_LOOKBACK,
    ) -> pd.DataFrame:
        """Fetch OHLCV as a DataFrame indexed by UTC datetime."""
        raw = await self._ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not raw:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df.drop(columns=["ts_ms"], inplace=True)
        df = df.astype(float)
        return df

    async def get_multi_tf_ohlcv(
        self, symbol: str, timeframes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple timeframes concurrently."""
        tfs = timeframes or TIMEFRAMES
        results = await asyncio.gather(
            *[self.get_ohlcv(symbol, tf) for tf in tfs],
            return_exceptions=True,
        )
        out: Dict[str, pd.DataFrame] = {}
        for tf, res in zip(tfs, results):
            if isinstance(res, Exception):
                logger.warning("OHLCV error %s %s: %s", symbol, tf, res)
                out[tf] = pd.DataFrame()
            else:
                out[tf] = res
        return out

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        ticker = await self._ex.fetch_ticker(symbol)
        return {
            "symbol":       symbol,
            "last":         ticker.get("last") or 0,
            "bid":          ticker.get("bid") or 0,
            "ask":          ticker.get("ask") or 0,
            "volume_24h":   ticker.get("quoteVolume") or 0,
            "change_pct":   ticker.get("percentage") or 0,
            "high_24h":     ticker.get("high") or 0,
            "low_24h":      ticker.get("low") or 0,
        }

    async def get_tickers_bulk(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch all tickers in one call (exchange supports it)."""
        all_tickers = await self._ex.fetch_tickers(symbols)
        out: Dict[str, Dict] = {}
        for sym, t in all_tickers.items():
            out[sym] = {
                "symbol":     sym,
                "last":       t.get("last") or 0,
                "volume_24h": t.get("quoteVolume") or 0,
                "change_pct": t.get("percentage") or 0,
                "high_24h":   t.get("high") or 0,
                "low_24h":    t.get("low") or 0,
            }
        return out

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    async def get_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, Any]:
        """Return raw order book dict with bids/asks as [[price, qty], ...]."""
        ob = await self._ex.fetch_order_book(symbol, limit=limit)
        return {
            "symbol":    symbol,
            "bids":      ob.get("bids", []),
            "asks":      ob.get("asks", []),
            "timestamp": ob.get("timestamp") or 0,
        }

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            info = await self._ex.fetch_funding_rate(symbol)
            return info.get("fundingRate")
        except Exception as exc:
            logger.debug("Funding rate error %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Recent trades
    # ------------------------------------------------------------------

    async def get_recent_trades(
        self, symbol: str, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Fetch recent public trades for order-flow bootstrapping."""
        trades = await self._ex.fetch_trades(symbol, limit=limit)
        return [
            {
                "ts":     t["timestamp"],
                "price":  float(t["price"]),
                "amount": float(t["amount"]),
                "side":   t["side"],  # 'buy' | 'sell'
            }
            for t in trades
        ]

    # ------------------------------------------------------------------
    # Historical data for backtesting
    # ------------------------------------------------------------------

    async def get_ohlcv_since(
        self,
        symbol: str,
        timeframe: str,
        since_iso: str,
        until_iso: Optional[str] = None,
    ) -> pd.DataFrame:
        """Paginated OHLCV fetch from `since_iso` date."""
        since_ms = int(
            datetime.fromisoformat(since_iso)
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        until_ms: Optional[int] = None
        if until_iso:
            until_ms = int(
                datetime.fromisoformat(until_iso)
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )

        all_rows: List[list] = []
        current = since_ms

        while True:
            batch = await self._ex.fetch_ohlcv(
                symbol, timeframe, since=current, limit=1000
            )
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            if until_ms and last_ts >= until_ms:
                break
            if len(batch) < 1000:
                break
            current = last_ts + 1
            await asyncio.sleep(0.2)  # respect rate limit

        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            all_rows, columns=["ts_ms", "open", "high", "low", "close", "volume"]
        )
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df.drop(columns=["ts_ms"], inplace=True)
        df = df.astype(float)

        if until_ms:
            df = df[df.index.astype("int64") // 10**6 <= until_ms]

        logger.info(
            "Fetched %d candles for %s %s (%s → %s)",
            len(df), symbol, timeframe, since_iso, until_iso or "now"
        )
        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_exchange(name: str = "binance") -> BaseExchange:
    """Return the exchange adapter for `name`. Extend for new venues."""
    registry = {
        "binance": BinanceFuturesExchange,
    }
    cls = registry.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown exchange '{name}'. Available: {list(registry)}")
    return cls()
