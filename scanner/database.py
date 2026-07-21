"""
Database module — async SQLite via aiosqlite.
Stores signals, OHLCV snapshots, order-flow events, and performance stats.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from .config import DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    pair            TEXT    NOT NULL,
    direction       TEXT    NOT NULL,  -- LONG | SHORT
    score           REAL    NOT NULL,
    entry_price     REAL,
    tp_price        REAL,
    sl_price        REAL,
    confirmations   TEXT,              -- JSON array
    raw_scores      TEXT,              -- JSON object
    status          TEXT    DEFAULT 'OPEN',  -- OPEN | WIN | LOSS | CANCELLED
    close_price     REAL,
    pnl_pct         REAL,
    closed_at       TEXT
);

CREATE TABLE IF NOT EXISTS ohlcv_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pair        TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    UNIQUE(pair, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS orderflow_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    pair        TEXT    NOT NULL,
    buy_vol     REAL,
    sell_vol    REAL,
    delta       REAL,
    delta_pct   REAL,
    event_type  TEXT    -- ABSORPTION_BULL | ABSORPTION_BEAR | IMBALANCE | SPIKE
);

CREATE TABLE IF NOT EXISTS performance_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at     TEXT NOT NULL,
    total_signals   INTEGER,
    wins            INTEGER,
    losses          INTEGER,
    win_rate        REAL,
    profit_factor   REAL,
    max_drawdown    REAL,
    avg_rr          REAL,
    avg_score       REAL
);

CREATE INDEX IF NOT EXISTS ix_signals_pair  ON signals (pair);
CREATE INDEX IF NOT EXISTS ix_signals_ts    ON signals (ts);
CREATE INDEX IF NOT EXISTS ix_ohlcv_pair_tf ON ohlcv_cache (pair, timeframe, ts);
"""


# ---------------------------------------------------------------------------
# DB manager
# ---------------------------------------------------------------------------

class Database:
    """Async SQLite wrapper.  Call await db.init() once before use."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        logger.info("Database initialised: %s", self.path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            logger.info("Database closed")

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def save_signal(
        self,
        pair: str,
        direction: str,
        score: float,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        confirmations: List[str],
        raw_scores: Dict[str, float],
    ) -> int:
        ts = datetime.utcnow().isoformat()
        async with self._conn.execute(
            """
            INSERT INTO signals
                (ts, pair, direction, score, entry_price, tp_price, sl_price,
                 confirmations, raw_scores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, pair, direction, score, entry_price, tp_price, sl_price,
                json.dumps(confirmations), json.dumps(raw_scores),
            ),
        ) as cur:
            row_id = cur.lastrowid
        await self._conn.commit()
        logger.debug("Signal saved id=%s pair=%s score=%.1f", row_id, pair, score)
        return row_id

    async def close_signal(
        self, signal_id: int, close_price: float, pnl_pct: float, status: str
    ) -> None:
        await self._conn.execute(
            """
            UPDATE signals
            SET close_price=?, pnl_pct=?, status=?, closed_at=?
            WHERE id=?
            """,
            (close_price, pnl_pct, status, datetime.utcnow().isoformat(), signal_id),
        )
        await self._conn.commit()

    async def get_open_signals(self) -> List[Dict]:
        async with self._conn.execute(
            "SELECT * FROM signals WHERE status='OPEN' ORDER BY ts DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_last_signal_ts(self, pair: str) -> Optional[str]:
        async with self._conn.execute(
            "SELECT ts FROM signals WHERE pair=? ORDER BY ts DESC LIMIT 1", (pair,)
        ) as cur:
            row = await cur.fetchone()
        return row["ts"] if row else None

    async def get_signals(
        self,
        pair: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict]:
        if pair:
            q = "SELECT * FROM signals WHERE pair=? ORDER BY ts DESC LIMIT ?"
            params = (pair, limit)
        else:
            q = "SELECT * FROM signals ORDER BY ts DESC LIMIT ?"
            params = (limit,)
        async with self._conn.execute(q, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # OHLCV cache
    # ------------------------------------------------------------------

    async def upsert_candles(
        self, pair: str, timeframe: str, rows: List[Dict]
    ) -> None:
        """Bulk insert/update OHLCV rows (idempotent)."""
        data = [
            (
                pair, timeframe,
                r["ts"], r["open"], r["high"], r["low"], r["close"], r["volume"],
            )
            for r in rows
        ]
        await self._conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv_cache
                (pair, timeframe, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        await self._conn.commit()

    async def get_candles(
        self, pair: str, timeframe: str, limit: int = 300
    ) -> List[Dict]:
        async with self._conn.execute(
            """
            SELECT * FROM ohlcv_cache
            WHERE pair=? AND timeframe=?
            ORDER BY ts DESC LIMIT ?
            """,
            (pair, timeframe, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Order flow
    # ------------------------------------------------------------------

    async def save_orderflow_event(
        self,
        pair: str,
        buy_vol: float,
        sell_vol: float,
        delta: float,
        delta_pct: float,
        event_type: str,
    ) -> None:
        await self._conn.execute(
            """
            INSERT INTO orderflow_events
                (ts, pair, buy_vol, sell_vol, delta, delta_pct, event_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                pair, buy_vol, sell_vol, delta, delta_pct, event_type,
            ),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Performance stats
    # ------------------------------------------------------------------

    async def compute_and_save_stats(self) -> Dict[str, Any]:
        async with self._conn.execute(
            "SELECT * FROM signals WHERE status IN ('WIN','LOSS')"
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

        if not rows:
            return {}

        total   = len(rows)
        wins    = sum(1 for r in rows if r["status"] == "WIN")
        losses  = total - wins
        win_rate = wins / total if total else 0

        gross_profit = sum(r["pnl_pct"] for r in rows if r["pnl_pct"] and r["pnl_pct"] > 0)
        gross_loss   = abs(sum(r["pnl_pct"] for r in rows if r["pnl_pct"] and r["pnl_pct"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

        # Max drawdown (equity curve based on sum of pnl_pct)
        equity = 0.0
        peak   = 0.0
        max_dd = 0.0
        for r in rows:
            equity += r["pnl_pct"] or 0
            peak = max(peak, equity)
            dd = (peak - equity) / (peak + 1e-9)
            max_dd = max(max_dd, dd)

        avg_rr    = sum(r["pnl_pct"] for r in rows if r["pnl_pct"]) / total
        avg_score = sum(r["score"] for r in rows) / total

        stats = {
            "total_signals": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "avg_rr": avg_rr,
            "avg_score": avg_score,
        }

        await self._conn.execute(
            """
            INSERT INTO performance_stats
                (computed_at, total_signals, wins, losses,
                 win_rate, profit_factor, max_drawdown, avg_rr, avg_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                stats["total_signals"], stats["wins"], stats["losses"],
                stats["win_rate"], stats["profit_factor"],
                stats["max_drawdown"], stats["avg_rr"], stats["avg_score"],
            ),
        )
        await self._conn.commit()
        return stats
