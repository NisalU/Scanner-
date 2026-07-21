"""
Main entry point — orchestrates all modules in a continuous async loop.
Run:  python -m scanner.main  or  python scanner/main.py
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

try:
    from .config import (
        ALERT_SCORE_THRESHOLD,
        BACKTEST_END,
        BACKTEST_START,
        BACKTEST_TIMEFRAME,
        CANDLE_LOOKBACK,
        MAX_PAIRS,
        SCAN_INTERVAL,
        TIMEFRAMES,
    )
    from .alerts import alert_manager
    from .backtest import run_backtest_async, print_report
    from .database import Database
    from .exchange import create_exchange, BinanceFuturesExchange
    from .indicators import add_all_indicators
    from .liquidity import analyse_order_book
    from .market_structure import analyse_structure
    from .orderflow import OrderFlowAggregator, analyse_orderflow
    from .scoring import determine_direction, score_signal
    from .websocket import BinanceFuturesWS, KlineStream, store as ws_store
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scanner.config import (
        ALERT_SCORE_THRESHOLD,
        BACKTEST_END,
        BACKTEST_START,
        BACKTEST_TIMEFRAME,
        CANDLE_LOOKBACK,
        MAX_PAIRS,
        SCAN_INTERVAL,
        TIMEFRAMES,
    )
    from scanner.alerts import alert_manager
    from scanner.backtest import run_backtest_async, print_report
    from scanner.database import Database
    from scanner.exchange import create_exchange, BinanceFuturesExchange
    from scanner.indicators import add_all_indicators
    from scanner.liquidity import analyse_order_book
    from scanner.market_structure import analyse_structure
    from scanner.orderflow import OrderFlowAggregator, analyse_orderflow
    from scanner.scoring import determine_direction, score_signal
    from scanner.websocket import BinanceFuturesWS, KlineStream, store as ws_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class CryptoScanner:
    def __init__(self) -> None:
        self.exchange = create_exchange("binance")
        self.db       = Database()
        self.of_agg   = OrderFlowAggregator(candle_seconds=60, max_history=200)
        self.symbols:  List[str] = []
        self._ws:       Optional[BinanceFuturesWS] = None
        self._kl:       Optional[KlineStream]       = None
        self._running   = False

        # Cache: symbol → timeframe → DataFrame
        self._ohlcv_cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("═" * 60)
        logger.info("  CryptoFutures Scanner — starting up")
        logger.info("═" * 60)

        await self.exchange.init()
        await self.db.init()

        # Universe
        self.symbols = await self.exchange.get_usdt_futures_pairs()
        logger.info("Scanning %d pairs", len(self.symbols))

        # Seed historical OHLCV
        await self._seed_ohlcv()

        # WebSocket streams
        self._ws = BinanceFuturesWS(self.symbols)
        ws_store.on_trade(self._on_ws_trade)
        await self._ws.start()

        self._kl = KlineStream(self.symbols, "1m")
        self._kl.on_candle_close(self._on_candle_close)
        await self._kl.start()

        self._running = True
        logger.info("Scanner ready — scanning every %ds (threshold: %d/100)",
                    SCAN_INTERVAL, ALERT_SCORE_THRESHOLD)

        await self._main_loop()

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.stop()
        if self._kl:
            await self._kl.stop()
        await self.exchange.close()
        await self.db.close()
        logger.info("Scanner stopped.")

    # ------------------------------------------------------------------
    # OHLCV seeding
    # ------------------------------------------------------------------

    async def _seed_ohlcv(self) -> None:
        logger.info("Seeding OHLCV for %d symbols × %d timeframes…", len(self.symbols), len(TIMEFRAMES))

        # Chunk to avoid rate-limit bursts
        chunk_size = 10
        for i in range(0, len(self.symbols), chunk_size):
            chunk = self.symbols[i : i + chunk_size]
            tasks = [self._fetch_and_cache(sym) for sym in chunk]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(1)

        logger.info("OHLCV seeding complete")

    async def _fetch_and_cache(self, symbol: str) -> None:
        try:
            dfs = await self.exchange.get_multi_tf_ohlcv(symbol)
            self._ohlcv_cache[symbol] = {}
            for tf, df in dfs.items():
                if not df.empty:
                    df = add_all_indicators(df)
                    self._ohlcv_cache[symbol][tf] = df
        except Exception as exc:
            logger.warning("Seed failed %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Real-time callbacks
    # ------------------------------------------------------------------

    async def _on_ws_trade(self, event: Dict) -> None:
        """Feed live trades to the order-flow aggregator."""
        symbol = event.get("symbol", "")
        self.of_agg.process_trade(
            symbol,
            int(event.get("ts", 0)),
            float(event.get("price", 0)),
            float(event.get("qty", 0)),
            str(event.get("side", "buy")),
        )

    async def _on_candle_close(self, candle: Dict) -> None:
        """Append new closed candle to cache and re-run indicators."""
        symbol = candle.get("symbol", "")
        tf     = candle.get("timeframe", "1m")

        if symbol not in self._ohlcv_cache:
            return

        new_row = pd.DataFrame([{
            "open":   candle["open"],
            "high":   candle["high"],
            "low":    candle["low"],
            "close":  candle["close"],
            "volume": candle["volume"],
        }], index=[pd.Timestamp(candle["ts"], unit="ms", tz="UTC")])

        cached = self._ohlcv_cache[symbol].get(tf)
        if cached is None or cached.empty:
            return

        updated = pd.concat([cached, new_row]).iloc[-CANDLE_LOOKBACK:]
        updated = add_all_indicators(updated)
        self._ohlcv_cache[symbol][tf] = updated

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        while self._running:
            start_ts = time.monotonic()
            try:
                await self._scan_all()
            except Exception as exc:
                logger.error("Main loop error: %s", exc, exc_info=True)

            elapsed = time.monotonic() - start_ts
            sleep_t = max(0, SCAN_INTERVAL - elapsed)
            logger.debug("Scan cycle took %.1fs, sleeping %.1fs", elapsed, sleep_t)
            await asyncio.sleep(sleep_t)

    async def _scan_all(self) -> None:
        """Evaluate every symbol and emit alerts where warranted."""
        logger.info("── Scan cycle starting (%d pairs) ──", len(self.symbols))
        tasks = [self._evaluate_symbol(sym) for sym in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        alerts_sent = sum(1 for r in results if r is True)
        logger.info("── Scan cycle done — %d alert(s) sent ──", alerts_sent)

    async def _evaluate_symbol(self, symbol: str) -> bool:
        """
        Full evaluation pipeline for one symbol.
        Returns True if an alert was fired.
        """
        try:
            cache = self._ohlcv_cache.get(symbol, {})
            df_4h  = cache.get("4h")
            df_1h  = cache.get("1h")
            df_15m = cache.get("15m")

            # Need all three higher-TF DataFrames
            for df in [df_4h, df_1h, df_15m]:
                if df is None or df.empty:
                    return False

            # Market structure
            struct_4h = analyse_structure(df_4h)
            struct_1h = analyse_structure(df_1h)

            # Direction
            direction = determine_direction(struct_4h, struct_1h, df_1h)
            if direction is None:
                return False

            # Current price
            last_price = float(df_1h["close"].iloc[-1])

            # Order flow
            of_candles = self.of_agg.get_candles(symbol)
            of_result  = analyse_orderflow(symbol, of_candles, df_15m)

            # Order book (fetch fresh from REST — lightweight)
            ob_result = None
            try:
                raw_ob = await self.exchange.get_order_book(symbol, limit=20)
                ob_result = analyse_order_book(
                    symbol, raw_ob["bids"], raw_ob["asks"]
                )
            except Exception as exc:
                logger.debug("Order book fetch failed %s: %s", symbol, exc)

            # Score
            signal = score_signal(
                symbol=symbol,
                direction=direction,
                price=last_price,
                df_4h=df_4h,
                df_1h=df_1h,
                df_15m=df_15m,
                struct_4h=struct_4h,
                struct_1h=struct_1h,
                of_result=of_result,
                ob_result=ob_result,
            )

            logger.debug(
                "%s  %s  score=%.1f  [t=%.0f ms=%.0f vwap=%.0f vol=%.0f of=%.0f v=%.0f]",
                symbol.replace("/USDT:USDT", "USDT"),
                direction,
                signal.score,
                signal.breakdown.trend,
                signal.breakdown.market_structure,
                signal.breakdown.vwap,
                signal.breakdown.volume,
                signal.breakdown.order_flow,
                signal.breakdown.volatility,
            )

            if signal.alert_worthy:
                # Persist
                await self.db.save_signal(
                    pair=symbol,
                    direction=direction,
                    score=signal.score,
                    entry_price=signal.entry_price,
                    tp_price=signal.tp_price,
                    sl_price=signal.sl_price,
                    confirmations=signal.confirmations,
                    raw_scores=signal.breakdown.as_dict(),
                )
                # Alert
                return await alert_manager.send_signal(signal)

        except Exception as exc:
            logger.warning("Eval error %s: %s", symbol, exc)

        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CryptoFutures Scanner")
    sub = parser.add_subparsers(dest="command")

    # scan (default)
    sub.add_parser("scan", help="Run the live scanner (default)")

    # backtest
    bt = sub.add_parser("backtest", help="Run backtesting on a symbol")
    bt.add_argument("--symbol", default="ETH/USDT:USDT", help="Symbol to backtest")
    bt.add_argument("--tf",     default=BACKTEST_TIMEFRAME, help="Timeframe (1h, 4h…)")
    bt.add_argument("--start",  default=BACKTEST_START,  help="Start date YYYY-MM-DD")
    bt.add_argument("--end",    default=BACKTEST_END,    help="End date YYYY-MM-DD")

    # stats
    sub.add_parser("stats", help="Show stored performance statistics")

    return parser.parse_args()


async def run_scan() -> None:
    scanner = CryptoScanner()
    loop = asyncio.get_running_loop()

    def _shutdown(sig):
        logger.info("Received signal %s — shutting down…", sig)
        loop.create_task(scanner.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except NotImplementedError:
            pass  # Windows

    await scanner.start()


async def run_backtest_cmd(args: argparse.Namespace) -> None:
    exchange = create_exchange("binance")
    await exchange.init()
    report = await run_backtest_async(
        exchange, args.symbol, args.tf, args.start, args.end
    )
    print_report(report)
    await exchange.close()


async def run_stats() -> None:
    db = Database()
    await db.init()
    stats = await db.compute_and_save_stats()
    if not stats:
        print("No closed signals in the database yet.")
    else:
        print("\n── Performance Statistics ──")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:<20}: {v:.4f}")
            else:
                print(f"  {k:<20}: {v}")
    await db.close()


def main() -> None:
    args = parse_args()
    cmd  = args.command or "scan"

    if cmd == "scan":
        asyncio.run(run_scan())
    elif cmd == "backtest":
        asyncio.run(run_backtest_cmd(args))
    elif cmd == "stats":
        asyncio.run(run_stats())
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
