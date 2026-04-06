# Trading Bots: End-to-End Guide — Analysis to Deployment

## The Pipeline

**Research → Backtest → Develop → Paper Trade → Deploy → Monitor**

Each stage has distinct tooling and concerns. Master them sequentially.

---

## Strategy Families

- **Arbitrage** — Price discrepancies across exchanges. Crypto is more viable due to fragmented markets. Includes triangular arb, cross-exchange arb, and DEX/CEX arb.
- **Market Making** — Posting both bid/ask and profiting from the spread. Requires fast execution and inventory management.
- **Momentum / Mean-Reversion** — Statistical strategies based on RSI, Bollinger bands, VWAP deviations, and other indicators.
- **Pairs Trading** — Cointegrated assets where you long one and short the other when they diverge.
- **Grid Trading** — Buy/sell orders at fixed intervals above and below a set price. Works well in ranging markets.

**Research tools:** Python with pandas, numpy, scipy. For crypto, **CCXT** is the universal exchange abstraction library (100+ exchanges).

---

## Backtesting Frameworks

- **Backtrader** (Python) — Mature, well-documented, supports multiple data feeds. Good starting point.
- **Vectorbt** — Vectorized backtesting, extremely fast for parameter sweeps.
- **Zipline / Zipline-reloaded** — Originally from Quantopian. More equity-focused.
- **FreqTrade** — Crypto-specific. Includes backtesting AND live trading in one package. Very strong choice.
- **Lean (QuantConnect)** — C#/Python, institutional-grade, free and open source.

> **⚠️ Backtesting Pitfalls:** Slippage modeling, transaction costs, look-ahead bias, survivorship bias, and overfitting. If your backtest shows >100% annual returns with low drawdown, you've almost certainly overfit.

---

## Hardware & Deployment

### Per-Bot Resource Footprint

- ~50–150MB RAM per bot, negligible CPU when idle
- WebSocket-based bots more efficient than REST polling
- ML-inference bots are the outlier — can consume GBs of VRAM

### Raspberry Pi 5 (16GB)

- Comfortably runs **20–50 lightweight bots**
- Use multiprocessing or asyncio (Python's GIL limits threading)
- Network I/O limits hit before RAM at 50+ bots

### 5090 Workstation (32GB RAM + GPU)

- **100 bots feasible** if most are lightweight
- GPU only matters for ML inference (transformers, LSTMs, RL agents)
- Pure rule-based strategies leave GPU idle

---

## Architecture for Scale

### Orchestrator Layer

- Manages bot lifecycle, restarts, and logging
- Tools: **Docker Compose**, **Supervisord**, or **PM2**

### Bot Pools

- **Pool A — Pi 5:** 20–50 lightweight bots
- **Pool B — Workstation:** 10–50 ML + heavy bots

### Shared Services

- **Redis** — State management and pub/sub messaging
- **PostgreSQL / TimescaleDB** — Trade logs and OHLCV data cache
- **Grafana** — Visual monitoring dashboards
- **Prometheus** — Metrics collection

---

## Execution & Integration

### Crypto

- **CCXT** is the standard (Binance, Kraken, Coinbase, Bybit, OKX, etc.)
- **FreqTrade** wraps CCXT with order management, DCA, trailing stops, stoploss
- For arb: need accounts on multiple exchanges with pre-funded balances

### Equities

- **Alpaca** — Commission-free, paper trading, best starting point for US equities
- **Interactive Brokers** via `ib_insync` — Most capable broker API, global coverage
- **Tradier**, **TD Ameritrade (Schwab)** — Alternatives

### Latency Considerations

- Pi on home WiFi adds ~20–50ms
- For crypto arb, colocation matters (e.g., AWS Tokyo for Binance)
- For swing/grid strategies, Pi latency is irrelevant

---

## Recommended Stack

1. **Start with FreqTrade** for crypto — backtesting, paper trading, and live trading in one tool
2. **Alpaca API** for equities
3. **Docker containers** per bot, **Docker Compose** or **k3s** for orchestration
4. **Redis** for shared state, **TimescaleDB** for time-series data

---

## Risk Management

> **🚨 Non-Negotiable Rules:**

- Never risk more than **1–2% of portfolio** per trade per bot
- Implement a **global kill switch** monitoring total portfolio drawdown
- **Paper trade 2–4 weeks minimum** before going live
- Start with **minimal capital** — scale only after consistent results
- **Rate limit awareness** — exchanges will ban excessive API calls. Use CCXT's built-in rate limiting.

---

## Realistic Expectations

Arbitrage opportunities in crypto have shrunk significantly since 2020–2021. Margins are thin and competition is fierce (colocated institutional firms). **Grid trading** and **mean-reversion** tend to be more accessible for retail traders. The edge for individuals is finding **niche pairs and markets** that institutional players ignore.
