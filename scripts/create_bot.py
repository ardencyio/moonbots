"""Scaffold a new bot from a template config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = {
    "bot_id": "{bot_id}",
    "strategy": "{strategy}",
    "ticker": "{ticker}",
    "timeframe": "1m",
    "market_session": {
        "open": "09:30:00",
        "close": "16:00:00",
        "timezone": "America/New_York",
    },
    "funds": {
        "account": "sub-account-{num}",
        "max_allocation_usd": 50000,
        "position_size_max_contracts": 2,
    },
    "risk_limits": {
        "max_daily_loss_usd": 1000,
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.05,
        "max_trades_per_day": 5,
        "circuit_breaker_after_loss": 3,
        "emergency_stop": True,
    },
    "parameters": {},
    "data_source": "yfinance",
    "execution": "paper",
    "poll_interval": 5,
}


def create_bot(bot_id: str, strategy: str, ticker: str) -> None:
    bot_dir = Path("bots") / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "state").mkdir(exist_ok=True)
    (bot_dir / "logs").mkdir(exist_ok=True)

    num = bot_id.split("-")[-1] if bot_id else "002"
    config = {
        **TEMPLATE,
        "bot_id": bot_id,
        "strategy": strategy,
        "ticker": ticker,
        "funds": {**TEMPLATE["funds"], "account": f"sub-account-{num}"},
    }

    (bot_dir / "bot.json").write_text(json.dumps(config, indent=2))
    print(f"Created bot: {bot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--strategy", default="open_gate")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    create_bot(args.id, args.strategy, args.ticker)
