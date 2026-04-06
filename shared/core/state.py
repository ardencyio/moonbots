"""State persistence for bots using SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional


@dataclass
class BotState:
    bot_id: str
    daily_start_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_losses: int = 0
    position_direction: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    todays_date: Optional[str] = None


class StateStore:
    """SQLite-backed state persistence for a bot."""

    def __init__(self, db_path: str | Path = "state/bot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create schema if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT,
                    timestamp TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    reason TEXT
                );
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    start_equity REAL,
                    end_equity REAL,
                    pnl REAL,
                    trades INTEGER,
                    losses INTEGER
                );
            """)

    def get(self, key: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT value FROM bot_state WHERE key = ?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value) VALUES (?, ?)",
                (key, value),
            )

    def record_trade(
        self,
        bot_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: Optional[float],
        pnl: float,
        reason: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                   (bot_id, timestamp, symbol, direction, entry_price, exit_price, pnl, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (bot_id, datetime.now().isoformat(), symbol, direction, entry_price, exit_price, pnl, reason),
            )

    def get_trades(self, bot_id: str, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE bot_id = ? ORDER BY timestamp DESC LIMIT ?",
                (bot_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_daily_stats(
        self,
        day: date,
        start_equity: float,
        end_equity: float,
        pnl: float,
        trades: int,
        losses: int,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_stats
                   (date, start_equity, end_equity, pnl, trades, losses)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (day.isoformat(), start_equity, end_equity, pnl, trades, losses),
            )
