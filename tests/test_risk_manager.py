"""Tests for RiskManager."""

import pytest

from shared.core.risk_manager import RiskManager


@pytest.fixture
def risk():
    return RiskManager(
        config={
            "max_daily_loss_usd": 1000,
            "max_daily_loss_pct": 0.02,
            "max_drawdown_pct": 0.05,
            "max_trades_per_day": 5,
            "max_position_size": 2,
            "circuit_breaker_after_loss": 3,
        },
        initial_capital=50_000,
    )


class TestPreTradeChecks:
    def test_allows_good_trade(self, risk):
        allowed, reasons = risk.check_pre_trade(potential_loss=100)
        assert allowed is True
        assert reasons == []

    def test_blocks_after_daily_loss(self, risk):
        risk.state.daily_pnl = -1000
        allowed, reasons = risk.check_pre_trade(potential_loss=100)
        assert allowed is False
        assert "Daily loss limit reached" in reasons

    def test_blocks_after_daily_loss_pct(self, risk):
        risk.state.daily_pnl = -1100
        allowed, reasons = risk.check_pre_trade(potential_loss=100)
        assert allowed is False

    def test_blocks_after_max_trades(self, risk):
        risk.state.daily_trades = 5
        allowed, reasons = risk.check_pre_trade(potential_loss=100)
        assert allowed is False
        assert "Max trades per day reached" in reasons

    def test_blocks_after_drawdown(self, risk):
        risk.state.max_drawdown = 0.055
        allowed, reasons = risk.check_pre_trade(potential_loss=100)
        assert allowed is False
        assert "Maximum drawdown limit reached" in reasons


class TestRecordTrade:
    def test_records_loss(self, risk):
        risk.record_trade(pnl=-500, is_loss=True)
        assert risk.state.daily_pnl == -500
        assert risk.state.consecutive_losses == 1

    def test_records_win(self, risk):
        risk.record_trade(pnl=500, is_loss=False)
        assert risk.state.daily_pnl == 500
        assert risk.state.consecutive_losses == 0

    def test_updates_drawdown(self, risk):
        risk.record_trade(pnl=-3000, is_loss=True)
        equity = 50_000 + risk.state.daily_pnl
        expected_dd = (50_000 - equity) / 50_000
        assert risk.state.max_drawdown == pytest.approx(expected_dd)


class TestStopTrading:
    def test_no_stop_when_healthy(self, risk):
        assert risk.should_stop_trading() is None

    def test_stops_at_daily_loss(self, risk):
        risk.state.daily_pnl = -1000
        reason = risk.should_stop_trading()
        assert reason is not None
        assert "Daily loss limit" in reason

    def test_circuit_breaker(self, risk):
        for _ in range(3):
            risk.record_trade(pnl=-200, is_loss=True)
        reason = risk.should_stop_trading()
        assert reason is not None
        assert "consecutive" in reason.lower()

    def test_drawdown_stop(self, risk):
        # Increase max_daily_loss_usd so drawdown triggers first
        risk.guardrails = risk.guardrails.__class__(max_daily_loss_usd=99999)
        risk.record_trade(pnl=-2600, is_loss=True)
        reason = risk.should_stop_trading()
        assert reason is not None
        assert "drawdown" in reason.lower()


class TestDailyReset:
    def test_reset_clears_daily_counters(self, risk):
        risk.state.daily_pnl = -500
        risk.state.daily_trades = 3
        risk.state.consecutive_losses = 2

        risk.reset_daily()

        assert risk.state.daily_pnl == 0
        assert risk.state.daily_trades == 0
        assert risk.state.consecutive_losses == 0
