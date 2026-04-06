"""Streamlit dashboard for monitoring trading bots."""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Bot Farm Dashboard", layout="wide")

st.title("Trading Bot Farm Dashboard")

# Show bot configs
bots_dir = Path("bots")
available_bots = sorted(
    [d for d in bots_dir.iterdir() if (d / "bot.json").exists()]
)

if available_bots:
    bot_names = [d.name for d in available_bots]
    selected = st.selectbox("Select Bot", bot_names)
    bot_dir = bots_dir / selected

    col1, col2, col3 = st.columns(3)
    with open(bot_dir / "bot.json") as f:
        config = json.load(f)

    with col1:
        st.metric("Strategy", config.get("strategy", "N/A"))
    with col2:
        st.metric("Ticker", config.get("ticker", "N/A"))
    with col3:
        st.metric("State DB", "Yes" if (bot_dir / "state" / "bot.db").exists() else "No")

    st.subheader("Risk Limits")
    limits = config.get("risk_limits", {})
    st.json(limits)

    st.subheader("Strategy Parameters")
    st.json(config.get("parameters", {}))

    # Trade history
    from shared.core.state import StateStore
    state_db = bot_dir / "state" / "bot.db"
    if state_db.exists():
        store = StateStore(str(state_db))
        trades = store.get_trades(selected)
        if trades:
            st.subheader("Recent Trades")
            for t in trades[:20]:
                st.write(f"{t['timestamp']} | {t['direction']} {t['symbol']} @ {t['entry_price']} | PnL: ${t['pnl']:.2f}")
else:
    st.warning("No bots configured. Create one with `scripts/create_bot.py`.")
