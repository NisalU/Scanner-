"""
Backtesting Module
Replays historical OHLCV data through the indicator + scoring pipeline
and measures strategy performance metrics.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_END,
    BACKTEST_START,
    BACKTEST_TIMEFRAME,
    CANDLE_LOOKBACK,
    REWARD_RISK_RATIO,
    RISK_PER_TRADE,
    SCORE_WEIGHTS,
    ALERT_SCORE_THRESHOLD,
    EMA_SLOW,
    ATR_PERIOD,
)
from .indicators import add_all_indicators
from .market_structure import analyse_structure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    idx:        int
    ts:         pd.Timestamp
    symbol:     str
    direction:  str
    entry:      float
    tp:         float
    sl:         float
    score:      float
    exit_price: float = 0.0
    exit_ts:    Optional[pd.Timestamp] = None
    result:     str   = "OPEN"   # WIN | LOSS | OPEN
    pnl_pct:    float = 0.0
    bars_held:  int   = 0


@dataclass
class BacktestReport:
    symbol:        str
    timeframe:     str
    start:         str
    end:           str
    total_trades:  int   = 0
    wins:          int   = 0
    losses:        int   = 0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    max_drawdown:  float = 0.0
    avg_rr:        float = 0.0
    sharpe:        float = 0.0
    total_return:  float = 0.0
    trades:        List[BacktestTrade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simple signal generator for backtesting
# (mirrors live logic without WS / orderflow dependencies)
# ---------------------------------------------------------------------------

def _generate_backtest_signal(
    df_window: pd.DataFrame,
) -> Optional[Tuple[str, float, float, float, float]]:
    """
    Lightweight signal generation for backtest.
    Returns (direction, score, entry, tp, sl) or None.
    Only uses indicator + structure data (no live order-flow).
    """
    if len(df_window) < EMA_SLOW + 20:
        return None

    df = add_all_indicators(df_window.copy())
    if df.empty:
        return None

    struct = analyse_structure(df)
    last   = df.iloc[-1]

    # Direction heuristic
    ema20_col = "ema_20"
    ema50_col = "ema_50"
    ema200_col = "ema_200"
    rsi_col   = f"rsi_{14}"
    atr_col   = f"atr_{ATR_PERIOD}"

    required = [ema20_col, ema50_col, ema200_col, rsi_col, atr_col, "vwap", "vol_ratio"]
    if not all(c in last.index for c in required):
        return None

    close  = float(last["close"])
    e20    = float(last[ema20_col])
    e50    = float(last[ema50_col])
    e200   = float(last[ema200_col])
    rsi_v  = float(last[rsi_col])
    atr_v  = float(last[atr_col])
    vwap_v = float(last["vwap"])
    vol_r  = float(last["vol_ratio"])

    # ---- LONG setup ----
    long_score = 0.0
    if close > e20 > e50 > e200:                   long_score += 30   # EMA bull stack
    if close > vwap_v:                              long_score += 15   # Above VWAP
    if 30 < rsi_v < 70:                             long_score += 10   # RSI healthy
    if struct.hh_hl:                                long_score += 20   # Market structure
    if struct.last_bos == "BOS_BULL":               long_score += 15   # BOS
    if vol_r >= 2.0:                                long_score += 10   # Volume

    # ---- SHORT setup ----
    short_score = 0.0
    if close < e20 < e50 < e200:                   short_score += 30
    if close < vwap_v:                              short_score += 15
    if 30 < rsi_v < 70:                             short_score += 10
    if struct.lh_ll:                                short_score += 20
    if struct.last_bos == "BOS_BEAR":               short_score += 15
    if vol_r >= 2.0:                                short_score += 10

    best_score = max(long_score, short_score)
    if best_score < ALERT_SCORE_THRESHOLD:
        return None

    direction = "LONG" if long_score >= short_score else "SHORT"
    entry = close

    if direction == "LONG":
        sl = entry - atr_v * 1.5
        tp = entry + atr_v * 3.0
    else:
        sl = entry + atr_v * 1.5
        tp = entry - atr_v * 3.0

    return direction, best_score, entry, tp, sl


# ---------------------------------------------------------------------------
# Trade resolver
# ---------------------------------------------------------------------------

def _resolve_trade(
    trade: BacktestTrade,
    future_df: pd.DataFrame,
    max_bars: int = 100,
) -> BacktestTrade:
    """
    Walk forward through `future_df` to check if TP or SL is hit first.
    """
    for i, (ts, row) in enumerate(future_df.iterrows()):
        if i >= max_bars:
            break
        high = float(row["high"])
        low  = float(row["low"])

        if trade.direction == "LONG":
            if low <= trade.sl:
                trade.result     = "LOSS"
                trade.exit_price = trade.sl
                trade.exit_ts    = ts
                trade.bars_held  = i + 1
                trade.pnl_pct    = (trade.sl - trade.entry) / trade.entry * 100
                return trade
            if high >= trade.tp:
                trade.result     = "WIN"
                trade.exit_price = trade.tp
                trade.exit_ts    = ts
                trade.bars_held  = i + 1
                trade.pnl_pct    = (trade.tp - trade.entry) / trade.entry * 100
                return trade
        else:  # SHORT
            if high >= trade.sl:
                trade.result     = "LOSS"
                trade.exit_price = trade.sl
                trade.exit_ts    = ts
                trade.bars_held  = i + 1
                trade.pnl_pct    = (trade.entry - trade.sl) / trade.entry * 100
                return trade
            if low <= trade.tp:
                trade.result     = "WIN"
                trade.exit_price = trade.tp
                trade.exit_ts    = ts
                trade.bars_held  = i + 1
                trade.pnl_pct    = (trade.entry - trade.tp) / trade.entry * 100
                return trade

    # Not resolved within max_bars → close at last price
    last_close = float(future_df["close"].iloc[-1]) if not future_df.empty else trade.entry
    trade.result     = "LOSS" if (
        (trade.direction == "LONG"  and last_close < trade.entry) or
        (trade.direction == "SHORT" and last_close > trade.entry)
    ) else "WIN"
    trade.exit_price = last_close
    trade.exit_ts    = future_df.index[-1] if not future_df.empty else None
    trade.bars_held  = len(future_df)
    trade.pnl_pct    = (
        (last_close - trade.entry) / trade.entry * 100
        if trade.direction == "LONG"
        else (trade.entry - last_close) / trade.entry * 100
    )
    return trade


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(trades: List[BacktestTrade]) -> Dict:
    closed = [t for t in trades if t.result in ("WIN", "LOSS")]
    if not closed:
        return {}

    total  = len(closed)
    wins   = sum(1 for t in closed if t.result == "WIN")
    losses = total - wins
    win_rate = wins / total

    pnls = [t.pnl_pct for t in closed]

    gross_profit = sum(p for p in pnls if p > 0) or 1e-10
    gross_loss   = abs(sum(p for p in pnls if p < 0)) or 1e-10
    profit_factor = gross_profit / gross_loss

    # Equity curve
    equity = [0.0]
    for p in pnls:
        equity.append(equity[-1] + p * RISK_PER_TRADE)

    peak   = 0.0
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd   = (peak - v) / (peak + 1e-10)
        max_dd = max(max_dd, dd)

    total_return = equity[-1]

    # Sharpe (daily-ish)
    returns = np.diff(equity)
    sharpe  = (returns.mean() / (returns.std() + 1e-10)) * (252 ** 0.5) if len(returns) > 1 else 0.0

    avg_rr  = np.mean(pnls) if pnls else 0.0
    avg_win  = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0.0
    avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0.0

    return {
        "total": total, "wins": wins, "losses": losses,
        "win_rate": win_rate, "profit_factor": profit_factor,
        "max_drawdown": max_dd, "total_return": total_return,
        "sharpe": sharpe, "avg_rr": avg_rr,
        "avg_win": avg_win, "avg_loss": avg_loss,
    }


# ---------------------------------------------------------------------------
# Main backtesting runner
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    timeframe: str = BACKTEST_TIMEFRAME,
    window: int = CANDLE_LOOKBACK,
    start: str = BACKTEST_START,
    end:   str = BACKTEST_END,
) -> BacktestReport:
    """
    Walk-forward backtest on `df`.
    Generates a new signal candidate every `step` bars,
    resolves against subsequent bars.
    """
    report = BacktestReport(
        symbol=symbol, timeframe=timeframe, start=start, end=end
    )

    if df.empty or len(df) < window + 10:
        logger.warning("Not enough data for backtest: %s (%d bars)", symbol, len(df))
        return report

    step = max(1, window // 10)
    trades: List[BacktestTrade] = []
    open_trade: Optional[BacktestTrade] = None

    for i in range(window, len(df) - 1, step):
        # Skip bar if we already have an open trade
        if open_trade and open_trade.result == "OPEN":
            continue

        df_window = df.iloc[i - window : i]
        result = _generate_backtest_signal(df_window)
        if result is None:
            continue

        direction, score, entry, tp, sl = result
        trade = BacktestTrade(
            idx=i,
            ts=df.index[i],
            symbol=symbol,
            direction=direction,
            entry=entry,
            tp=tp,
            sl=sl,
            score=score,
        )

        future_df = df.iloc[i + 1 : i + 101]
        trade = _resolve_trade(trade, future_df)
        trades.append(trade)
        open_trade = trade

    report.trades = trades
    metrics = _compute_metrics(trades)

    report.total_trades  = metrics.get("total", 0)
    report.wins          = metrics.get("wins", 0)
    report.losses        = metrics.get("losses", 0)
    report.win_rate      = metrics.get("win_rate", 0.0)
    report.profit_factor = metrics.get("profit_factor", 0.0)
    report.max_drawdown  = metrics.get("max_drawdown", 0.0)
    report.avg_rr        = metrics.get("avg_rr", 0.0)
    report.sharpe        = metrics.get("sharpe", 0.0)
    report.total_return  = metrics.get("total_return", 0.0)

    return report


def print_report(report: BacktestReport) -> None:
    """Pretty-print backtest report to stdout."""
    sym = report.symbol.replace("/USDT:USDT", "USDT")
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT — {sym}  [{report.timeframe}]")
    print(f"  Period: {report.start} → {report.end}")
    print(sep)
    print(f"  Total Trades : {report.total_trades}")
    print(f"  Wins         : {report.wins}")
    print(f"  Losses       : {report.losses}")
    print(f"  Win Rate     : {report.win_rate * 100:.1f}%")
    print(f"  Profit Factor: {report.profit_factor:.2f}")
    print(f"  Max Drawdown : {report.max_drawdown * 100:.1f}%")
    print(f"  Total Return : {report.total_return:.2f}%  (at {RISK_PER_TRADE*100:.0f}% risk/trade)")
    print(f"  Sharpe Ratio : {report.sharpe:.2f}")
    print(f"  Avg P&L/trade: {report.avg_rr:.2f}%")
    print(sep)

    if report.trades:
        print("  Last 10 Trades:")
        for t in report.trades[-10:]:
            icon = "✅" if t.result == "WIN" else "❌"
            ts_str = t.ts.strftime("%Y-%m-%d %H:%M") if t.ts else "N/A"
            print(
                f"    {icon} {ts_str}  {t.direction:<5}  "
                f"entry={t.entry:.4g}  pnl={t.pnl_pct:+.2f}%  "
                f"score={t.score:.0f}  bars={t.bars_held}"
            )
    print(sep)


# ---------------------------------------------------------------------------
# Async wrapper for use from main.py
# ---------------------------------------------------------------------------

async def run_backtest_async(
    exchange,
    symbol: str,
    timeframe: str = BACKTEST_TIMEFRAME,
    start: str = BACKTEST_START,
    end:   str = BACKTEST_END,
) -> BacktestReport:
    """Fetch historical data and run backtest."""
    logger.info("Fetching historical data for backtest: %s %s", symbol, timeframe)
    df = await exchange.get_ohlcv_since(symbol, timeframe, start, end)
    if df.empty:
        logger.error("No historical data returned for %s", symbol)
        return BacktestReport(symbol=symbol, timeframe=timeframe, start=start, end=end)

    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(
        None, run_backtest, symbol, df, timeframe, CANDLE_LOOKBACK, start, end
    )
    return report
