"""Reset daily stats at the start of each trading day."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def reset_daily_stats(bot_id: str = "nq-open-gate-001") -> None:
    db = Path(f"bots/{bot_id}/state/bot.db")
    if not db.exists():
        print(f"No state DB for {bot_id}")
        return

    conn = sqlite3.connect(db)
    # Clear the current day row if it exists
    from datetime import date

    today = date.today().isoformat()
    conn.execute("DELETE FROM daily_stats WHERE date = ?", (today,))
    conn.commit()
    print(f"Reset daily stats for {bot_id} ({today})")
    conn.close()


if __name__ == "__main__":
    reset_daily_stats()
