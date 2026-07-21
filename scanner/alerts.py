"""
Alert System — Telegram notifications.
Formats and sends trading signal alerts.
Respects per-pair cooldown to avoid spam.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Optional

import httpx

from .config import (
    ALERT_COOLDOWN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TOKEN,
)
from .scoring import SignalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

_TG_BASE = "https://api.telegram.org/bot{token}/{method}"


async def _tg_send(text: str, parse_mode: str = "HTML") -> bool:
    """Low-level Telegram sendMessage with retry."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID missing)")
        return False

    url = _TG_BASE.format(token=TELEGRAM_TOKEN, method="sendMessage")
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                data = resp.json()
                if data.get("ok"):
                    return True
                logger.error("Telegram API error: %s", data)
                return False
        except Exception as exc:
            logger.warning("Telegram send attempt %d failed: %s", attempt + 1, exc)
            await asyncio.sleep(2 ** attempt)
    return False


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------

def _direction_emoji(direction: str) -> str:
    return "🟢" if direction == "LONG" else "🔴"


def _score_bar(score: float, width: int = 10) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def format_signal_message(signal: SignalResult) -> str:
    """Format a full signal alert in HTML for Telegram."""
    emoji = _direction_emoji(signal.direction)
    bar   = _score_bar(signal.score)
    ts    = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Symbol display: strip the ccxt suffix for readability
    sym_display = signal.symbol.replace("/USDT:USDT", "USDT")

    conf_lines = "\n".join(f"  • {c}" for c in signal.confirmations[:8])

    bd = signal.breakdown
    score_lines = (
        f"  Trend .............. {bd.trend:.0f}/{bd.trend + 0:.0f}→{20}\n"
        f"  Market Structure ... {bd.market_structure:.0f}/20\n"
        f"  VWAP / Indicators .. {bd.vwap:.0f}/15\n"
        f"  Volume ............. {bd.volume:.0f}/15\n"
        f"  Order Flow ......... {bd.order_flow:.0f}/25\n"
        f"  Volatility ......... {bd.volatility:.0f}/5"
    )

    msg = (
        f"<b>{emoji} SIGNAL ALERT — {sym_display}</b>\n"
        f"<code>─────────────────────────────</code>\n"
        f"<b>DIRECTION:</b>  {signal.direction}\n"
        f"<b>SCORE:</b>      {signal.score:.0f}/100  [{bar}]\n"
        f"<code>─────────────────────────────</code>\n"
        f"<b>ENTRY AREA:</b>\n"
        f"  {signal.entry_price:.6g}\n\n"
        f"<b>TARGET (TP):</b>\n"
        f"  {signal.tp_price:.6g}\n\n"
        f"<b>INVALIDATION (SL):</b>\n"
        f"  {signal.sl_price:.6g}\n"
        f"<code>─────────────────────────────</code>\n"
        f"<b>CONFIRMATIONS:</b>\n{conf_lines}\n"
        f"<code>─────────────────────────────</code>\n"
        f"<b>SCORE BREAKDOWN:</b>\n<code>{score_lines}</code>\n"
        f"<code>─────────────────────────────</code>\n"
        f"<i>{ts}</i>"
    )
    return msg


def format_stats_message(stats: Dict) -> str:
    """Format performance statistics summary."""
    if not stats:
        return "<b>📊 No closed signals yet.</b>"

    wr  = stats.get("win_rate", 0) * 100
    pf  = stats.get("profit_factor", 0)
    mdd = stats.get("max_drawdown", 0) * 100
    ts  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"<b>📊 PERFORMANCE STATISTICS</b>\n"
        f"<code>─────────────────────────────</code>\n"
        f"<b>Total Signals:</b>  {stats.get('total_signals', 0)}\n"
        f"<b>Wins:</b>           {stats.get('wins', 0)}\n"
        f"<b>Losses:</b>         {stats.get('losses', 0)}\n"
        f"<b>Win Rate:</b>       {wr:.1f}%\n"
        f"<b>Profit Factor:</b>  {pf:.2f}\n"
        f"<b>Max Drawdown:</b>   {mdd:.1f}%\n"
        f"<b>Avg R/R:</b>        {stats.get('avg_rr', 0):.2f}\n"
        f"<b>Avg Score:</b>      {stats.get('avg_score', 0):.1f}\n"
        f"<code>─────────────────────────────</code>\n"
        f"<i>{ts}</i>"
    )


# ---------------------------------------------------------------------------
# Alert manager (cooldown-aware)
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Sends Telegram alerts with per-pair cooldown.
    Call `await manager.send_signal(signal)` from the scanner.
    """

    def __init__(self, cooldown: int = ALERT_COOLDOWN) -> None:
        self._cooldown = cooldown
        self._last_alert: Dict[str, float] = {}  # symbol → unix ts

    def _is_cooled_down(self, symbol: str) -> bool:
        last = self._last_alert.get(symbol, 0)
        return (time.time() - last) >= self._cooldown

    async def send_signal(self, signal: SignalResult) -> bool:
        """
        Format and send a signal alert.
        Returns True if the message was sent (or would have been if Telegram configured).
        Respects cooldown per symbol.
        """
        if not signal.alert_worthy:
            logger.debug(
                "Signal %s score %.1f below threshold — not alerting",
                signal.symbol, signal.score
            )
            return False

        if not self._is_cooled_down(signal.symbol):
            remaining = int(
                self._cooldown - (time.time() - self._last_alert.get(signal.symbol, 0))
            )
            logger.debug(
                "Alert for %s suppressed by cooldown (%ds remaining)",
                signal.symbol, remaining
            )
            return False

        msg = format_signal_message(signal)

        # Always log the alert to console regardless of Telegram config
        sym = signal.symbol.replace("/USDT:USDT", "USDT")
        logger.info(
            "🚨 ALERT  %s  %s  score=%.0f/100  entry=%.6g  tp=%.6g  sl=%.6g",
            sym, signal.direction, signal.score,
            signal.entry_price, signal.tp_price, signal.sl_price
        )
        logger.info("   Confirmations: %s", " | ".join(signal.confirmations[:5]))

        sent = await _tg_send(msg)
        if sent:
            self._last_alert[signal.symbol] = time.time()
            logger.info("Telegram alert sent for %s", sym)
        else:
            logger.warning("Telegram alert FAILED for %s", sym)
            # Still mark to avoid hammering if Telegram is down
            self._last_alert[signal.symbol] = time.time()

        return True

    async def send_stats(self, stats: Dict) -> bool:
        msg = format_stats_message(stats)
        logger.info("Sending performance stats via Telegram")
        return await _tg_send(msg)

    async def send_text(self, text: str) -> bool:
        return await _tg_send(f"<code>{text}</code>")


# Module-level singleton
alert_manager = AlertManager()
