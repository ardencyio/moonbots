"""Metrics collection for trading bots."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class BotMetrics:
    bot_id: str
    timestamp: datetime
    total_return: float = 0.0
    daily_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    today_trades: int = 0
    open_positions: int = 0
    current_drawdown: float = 0.0
    daily_loss: float = 0.0
    daily_loss_pct: float = 0.0


class MetricsCollector:
    """Collects and exposes bot performance metrics."""

    def __init__(self):
        self._latest: dict[str, BotMetrics] = {}

    def record(self, bot_id: str, metrics: BotMetrics) -> None:
        self._latest[bot_id] = metrics

    def get(self, bot_id: str) -> Optional[BotMetrics]:
        return self._latest.get(bot_id)

    def summary(self, bot_id: str) -> dict:
        m = self._latest.get(bot_id)
        if m is None:
            return {}
        return {
            "bot_id": m.bot_id,
            "timestamp": m.timestamp.isoformat(),
            "total_return_pct": m.total_return * 100,
            "sharpe_ratio": m.sharpe_ratio,
            "max_drawdown_pct": m.max_drawdown * 100,
            "total_trades": m.total_trades,
            "win_rate_pct": m.win_rate * 100,
        }

    def all_bots(self) -> dict[str, dict]:
        return {bot_id: self.summary(bot_id) for bot_id in self._latest}
