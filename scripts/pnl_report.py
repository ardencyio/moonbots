"""Generate P&L report from bot state files."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def pnl_report(bot_id: str = "nq-open-gate-001") -> None:
    db = Path(f"bots/{bot_id}/state/bot.db")
    if not db.exists():
        print(f"No state DB found for {bot_id}")
        return

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    # Daily stats
    rows = conn.execute(
        "SELECT * FROM daily_stats ORDER BY date DESC LIMIT 30"
    ).fetchall()

    print(f"=== P&L Report: {bot_id} ===")
    if not rows:
        print("No trade data recorded yet.")
        conn.close()
        return

    total_pnl = sum(r["pnl"] for r in rows)
    total_trades = sum(r["trades"] for r in rows)

    print(f"Days covered: {len(rows)}")
    print(f"Total P&L:    ${total_pnl:.2f}")
    print(f"Total trades: {total_trades}")
    print()

    for r in rows:
        print(f"  {r['date']}: pnl=${r['pnl']:.2f} trades={r['trades']} losses={r['losses']}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", default="nq-open-gate-001")
    pnl_report(parser.parse_args().bot)
