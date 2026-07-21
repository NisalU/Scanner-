# 🔍 CryptoFutures Scanner

A real-time, AI-assisted scanner for Binance USDT-M perpetual futures.  
Monitors hundreds of pairs simultaneously and fires high-probability alerts via Telegram.

---

## Features

| Module | What it does |
|---|---|
| **Multi-TF Analysis** | 4H bias → 1H zones → 15M setups → 5M/1M entry |
| **Market Structure** | HH/HL/LH/LL, Break of Structure, Change of Character, S/R zones |
| **Indicators** | VWAP, EMA 20/50/200, RSI, ATR, MACD, Bollinger Bands, Volume MA |
| **Order Flow** | Buy/sell delta, imbalance, bullish/bearish absorption, climax volume |
| **Liquidity** | Order-book walls, clusters, possible spoofing detection |
| **Scoring** | 0-100 composite score with 6 weighted dimensions |
| **Alerts** | Telegram with entry/TP/SL and full confirmation list |
| **Database** | SQLite — signals, OHLCV, order-flow events, performance stats |
| **Backtesting** | Walk-forward with win rate, profit factor, max drawdown, Sharpe |

---

## Project Structure

```
scanner/
├── __init__.py          package init
├── config.py            all settings loaded from .env
├── exchange.py          Binance Futures REST via ccxt (modular base class)
├── websocket.py         real-time aggTrade + depth streams (auto-reconnect)
├── indicators.py        VWAP, EMA, RSI, ATR, MACD, BB, Volume MA
├── market_structure.py  swing detection, BOS/CHOCH, S/R/liquidity zones
├── orderflow.py         delta, imbalance, absorption, climax volume
├── liquidity.py         order-book walls, clusters, spoof detection
├── scoring.py           0-100 composite scoring engine
├── alerts.py            Telegram formatter + cooldown-aware sender
├── database.py          async SQLite via aiosqlite
├── backtest.py          walk-forward backtester + metrics
└── main.py              CLI entry point (scan | backtest | stats)
```

---

## Quick Start (local)

### 1. Prerequisites

- Python 3.12+
- pip / venv

### 2. Clone / download

```bash
git clone <repo-url>
cd <repo-dir>
```

### 3. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure

```bash
cp scanner/.env.example scanner/.env
# Edit scanner/.env — add your Telegram token and chat ID at minimum.
# Binance API keys are only needed for private data; scanning is public.
```

### 6. Run the scanner

```bash
python -m scanner.main scan
```

Or directly:

```bash
python scanner/main.py scan
```

---

## Commands

### `scan` — live scanner (default)

```bash
python -m scanner.main scan
```

Starts the full pipeline:
1. Fetches top-volume USDT-M pairs (up to `MAX_PAIRS`)
2. Seeds 300 candles per timeframe (4H, 1H, 15M, 5M, 1M)
3. Opens WebSocket streams for real-time trades and order book
4. Evaluates every pair every `SCAN_INTERVAL` seconds
5. Sends Telegram alerts for signals scoring ≥ `ALERT_SCORE_THRESHOLD`

### `backtest` — historical strategy evaluation

```bash
python -m scanner.main backtest \
  --symbol ETH/USDT:USDT \
  --tf 1h \
  --start 2024-01-01 \
  --end 2025-01-01
```

Walk-forward backtest on historical data. Reports:
- Win rate, profit factor, max drawdown
- Sharpe ratio, average R/R
- Last 10 trades detail

### `stats` — performance report from live signals

```bash
python -m scanner.main stats
```

Computes and prints performance statistics from all closed signals in the database.

---

## Telegram Setup

1. Create a bot: message `@BotFather` on Telegram → `/newbot`
2. Copy the token into `TELEGRAM_TOKEN` in `.env`
3. Start a conversation with your bot, then run:

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
```

4. Find `"chat":{"id":XXXXXXXX}` → copy into `TELEGRAM_CHAT_ID` in `.env`

### Alert Format

```
🟢 SIGNAL ALERT — ETHUSDT
─────────────────────────────
DIRECTION:  LONG
SCORE:      87/100  [████████░░]
─────────────────────────────
ENTRY AREA:
  2341.50

TARGET (TP):
  2412.30

INVALIDATION (SL):
  2305.80
─────────────────────────────
CONFIRMATIONS:
  • 4H STRONG_BULL trend (EMA stack)
  • 1H HH+HL sequence
  • Price above 1H VWAP
  • Volume spike: 3.2× average
  • Bullish absorption detected
─────────────────────────────
SCORE BREAKDOWN:
  Trend .............. 18/20
  Market Structure ... 17/20
  VWAP / Indicators .. 12/15
  Volume ............. 14/15
  Order Flow ......... 21/25
  Volatility .........  5/5
─────────────────────────────
2025-01-15 08:42 UTC
```

---

## Scoring System

| Dimension | Weight | Signals |
|---|---|---|
| Trend | 20 | EMA stack alignment on 4H + 1H |
| Market Structure | 20 | HH/HL, BOS, CHOCH, zone touches |
| VWAP / Indicators | 15 | VWAP position, MACD cross, RSI |
| Volume | 15 | Spike multiplier, breakout volume |
| Order Flow | 25 | Absorption, delta, imbalance, OB bias |
| Volatility | 5 | ATR expansion, BB width |

Only signals scoring **≥ 80** generate alerts (configurable via `ALERT_SCORE_THRESHOLD`).

---

## Configuration Reference

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `BINANCE_API_KEY` | — | Optional; public endpoints work without it |
| `TELEGRAM_TOKEN` | — | Required for alerts |
| `TELEGRAM_CHAT_ID` | — | Required for alerts |
| `MIN_VOLUME_USDT` | 10,000,000 | 24h min volume to include a pair |
| `MAX_PAIRS` | 100 | Max pairs to scan |
| `SCAN_INTERVAL` | 60 | Seconds between scan cycles |
| `ALERT_SCORE_THRESHOLD` | 80 | Min score for alert |
| `ALERT_COOLDOWN` | 3600 | Seconds between alerts per pair |
| `CANDLE_LOOKBACK` | 300 | Candles to seed per timeframe |
| `BACKTEST_START` | 2024-01-01 | Default backtest start |
| `BACKTEST_END` | 2025-01-01 | Default backtest end |
| `LOG_LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR |

---

## Adding a New Exchange

1. Subclass `BaseExchange` in `scanner/exchange.py`
2. Implement the 6 abstract methods (`init`, `close`, `get_usdt_futures_pairs`, `get_ohlcv`, `get_ticker`, `get_order_book`, `get_funding_rate`)
3. Register it in the `create_exchange()` factory at the bottom of `exchange.py`
4. Update `BINANCE_*` env vars or add new ones in `config.py`

Example skeleton:

```python
class BybitFuturesExchange(BaseExchange):
    def __init__(self):
        self._ex = ccxt.async_support.bybit({"options": {"defaultType": "linear"}})
    # ... implement abstract methods ...

# In create_exchange():
registry = {
    "binance": BinanceFuturesExchange,
    "bybit":   BybitFuturesExchange,   # ← add here
}
```

---

## VPS Deployment (optional)

If you want 24/7 uptime on a remote server:

### Recommended specs
- 2 vCPU, 2 GB RAM minimum
- Ubuntu 22.04 or Debian 12

### Setup

```bash
# Install Python 3.12
sudo apt update && sudo apt install -y python3.12 python3.12-venv git screen

# Clone
git clone <repo> /opt/cryptoscanner
cd /opt/cryptoscanner

# Venv + deps
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configure
cp .env.example .env && nano .env

# Run in a persistent screen session
screen -S scanner
.venv/bin/python -m scanner.main scan
# Ctrl+A, D to detach
```

### systemd service (recommended over screen)

```ini
# /etc/systemd/system/cryptoscanner.service
[Unit]
Description=CryptoFutures Scanner
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/cryptoscanner
EnvironmentFile=/opt/cryptoscanner/.env
ExecStart=/opt/cryptoscanner/.venv/bin/python -m scanner.main scan
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cryptoscanner
sudo journalctl -fu cryptoscanner   # live logs
```

---

## Notes

- **Public endpoints** — scanning requires no API keys. Keys only add rate-limit headroom.
- **WebSocket reconnect** — streams reconnect automatically with exponential back-off.
- **Order-flow bootstrap** — on startup, recent REST trades seed the aggregator; live WS fills it from there.
- **Rate limits** — `ccxt` handles Binance rate limits automatically. With `MAX_PAIRS=100`, seeding takes ~30 s.
- **Spoof detection** — compares consecutive order-book snapshots; alerts are logged but do not affect scoring directly.
