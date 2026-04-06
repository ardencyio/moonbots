"""Bot entry point for nq-open-gate-001.

Usage:
    uv run -m bots.nq_open_gate_001.main [--config bots/nq_open_gate_001/bot.json]
    varlock run -- uv run -m bots.nq_open_gate_001.main [--config bots/nq_open_gate_001/bot.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, time
from pathlib import Path

from backtest.strategies.open_gate import OpenGateStrategy
from shared.core.risk_manager import RiskManager
from shared.core.execution import PaperExecutionHandler
from shared.core.state import StateStore
from shared.core.alerts import AlertManager, Alert, AlertSeverity
from scripts.env_loader import get_env


async def main(config_path: str) -> None:
    path = Path(config_path)
    with open(path) as f:
        config = json.load(f)

    bot_id = config["bot_id"]

    # Load sensitive env vars from varlock/1Password
    polygon_key = get_env("POLYGON_API_KEY")

    # Get market data API key from config or env
    data_source = config.get("data_source", "yfinance")
    if data_source == "polygon" and not polygon_key:
        raise RuntimeError("POLYGON_API_KEY required for Polygon data source")

    _ = StateStore(db_path=str(path.parent / "state" / "bot.db"))
    params = config.get("parameters", {})
    strategy = OpenGateStrategy(params)
    risk = RiskManager(
        config=config.get("risk_limits", {}),
        initial_capital=config.get("funds", {}).get("max_allocation_usd", 50_000),
    )
    execution = PaperExecutionHandler()
    alerts = AlertManager()

    # If real execution is configured, connect with real API keys
    execution_mode = config.get("execution", "paper")
    if execution_mode == "live" and data_source == "polygon":
        from backtest.data.polygon_fetcher import PolygonFetcher
        if polygon_key:
            execution = PolygonFetcher(polygon_key)

    await execution.connect()
    strategy.reset()
    risk.reset_daily()

    ticker = config.get("ticker", "NQ=F")
    poll_interval = config.get("poll_interval", 5)
    market_open = time.fromisoformat(config["market_session"]["open"])
    market_close = time.fromisoformat(config["market_session"]["close"])

    print(f"[{bot_id}] Bot started ({execution_mode} mode) — {ticker}")

    while True:
        now = datetime.now().time()
        if not (market_open <= now <= market_close):
            await asyncio.sleep(60)
            continue

        stop_reason = risk.should_stop_trading()
        if stop_reason:
            alerts.raise_alert(
                Alert(bot_id, AlertSeverity.CRITICAL, "Risk stop", stop_reason)
            )
            print(f"[{bot_id}] Trading stopped: {stop_reason}")
            await asyncio.sleep(300)
            continue

        try:
            await execution.get_account_balance()  # placeholder
            print(f"[{bot_id}] Heartbeat — PnL: {risk.state.daily_pnl:.2f}")
        except Exception as e:
            alerts.raise_alert(
                Alert(bot_id, AlertSeverity.WARNING, "Loop error", str(e))
            )

        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bots/nq_open_gate_001/bot.json")
    args = parser.parse_args()
    asyncio.run(main(args.config))
