"""Bot orchestrator — manages multiple bot processes and global risk."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from shared.core.risk_manager import RiskGuardrails


class BotProcess:
    def __init__(self, bot_id: str, config_path: Path):
        self.bot_id = bot_id
        self.config_path = config_path
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "bots.nq-open-gate-001.main", "--config", str(self.config_path)],
        )

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            self.process = None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


class GlobalRiskMonitor:
    def __init__(self, guardrails: Optional[RiskGuardrails] = None):
        self.guardrails = guardrails or RiskGuardrails()

    def check_global_limits(self, bots: list[BotProcess]) -> bool:
        """Return True if global limit is breached and all bots should stop."""
        dead = [b.bot_id for b in bots if not b.alive]
        if len(dead) > 0 and len(bots) > 0 and len(dead) / len(bots) > 0.5:
            return True
        return False


class BotOrchestrator:
    def __init__(self, config_dir: str | Path = "bots"):
        self.config_dir = Path(config_dir)
        self.bots: dict[str, BotProcess] = {}
        self.risk_monitor = GlobalRiskMonitor()

    def spawn_bot(self, bot_id: str) -> None:
        config = self.config_dir / bot_id / "bot.json"
        if not config.exists():
            raise FileNotFoundError(f"No bot config: {config}")
        self.bots[bot_id] = BotProcess(bot_id, config)
        self.bots[bot_id].start()

    def stop_bot(self, bot_id: str) -> None:
        bot = self.bots.get(bot_id)
        if bot:
            bot.stop()
            del self.bots[bot_id]

    def emergency_stop_all(self) -> None:
        for bot in list(self.bots.values()):
            bot.stop()
        self.bots.clear()

    def monitor(self) -> None:
        if self.risk_monitor.check_global_limits(list(self.bots.values())):
            self.emergency_stop_all()
