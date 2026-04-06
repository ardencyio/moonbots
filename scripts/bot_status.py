"""Check the health of running bot processes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def check_health(bot_dir: Path) -> dict:
    config = json.loads((bot_dir / "bot.json").read_text())
    bot_id = config["bot_id"]
    state_db = bot_dir / "state" / "bot.db"

    status = {
        "bot_id": bot_id,
        "timestamp": datetime.now().isoformat(),
        "config_ok": (bot_dir / "bot.json").exists(),
        "state_ok": state_db.exists(),
        "logs_ok": (bot_dir / "logs").exists(),
    }
    return status


if __name__ == "__main__":
    bots_dir = Path("bots")
    for d in bots_dir.iterdir():
        if d.is_dir() and (d / "bot.json").exists():
            s = check_health(d)
            ok = "✅" if all(v for v in s.values() if isinstance(v, bool)) else "❌"
            print(f"{ok} {s['bot_id']}: config={s['config_ok']} state={s['state_ok']}")
