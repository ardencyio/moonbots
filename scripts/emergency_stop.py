"""Emergency stop script — kill all bots or a specific bot."""

from __future__ import annotations

import argparse
import subprocess


def stop_all() -> None:
    """Kill all bot processes by name."""
    print("Stopping all bots...")
    result = subprocess.run(
        ["pkill", "-f", "bots.*main.py"],
        capture_output=True,
    )
    print(f"Done. exit={result.returncode}")


def stop_bot(bot_id: str) -> None:
    print(f"Stopping bot {bot_id}...")
    subprocess.run(
        ["pkill", "-f", bot_id],
        capture_output=True,
    )
    print(f"Bot {bot_id} stopped.")


def main():
    parser = argparse.ArgumentParser(description="Emergency stop for bots")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Stop all bots")
    group.add_argument("--bot", type=str, help="Stop a specific bot by ID")
    args = parser.parse_args()

    if args.all:
        stop_all()
    elif args.bot:
        stop_bot(args.bot)


if __name__ == "__main__":
    main()
