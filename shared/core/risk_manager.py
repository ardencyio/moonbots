"""Risk management for individual bots."""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class RiskGuardrails:
    """Hard-coded risk limits that cannot be overridden by bot logic."""

    # Daily loss limits
    max_daily_loss_usd: float = 1000.0
    max_daily_loss_pct: float = 0.02

    # Drawdown limits
    max_drawdown_pct: float = 0.05

    # Trade frequency
    max_trades_per_day: int = 5
    min_time_between_trades: int = 300  # seconds

    # Position sizing
    max_position_size: float = 2
    max_notional_exposure: float = 50_000

    # Emergency controls
    emergency_stop: bool = True
    circuit_breaker_after_loss: int = 3


@dataclass
class RiskState:
    daily_start_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_losses: int = 0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    todays_date: Optional[date] = None


class RiskManager:
    """Enforces risk limits for a single bot."""

    def __init__(self, config: dict, initial_capital: float):
        self.guardrails = RiskGuardrails(
            max_daily_loss_usd=config.get("max_daily_loss_usd", 1000.0),
            max_daily_loss_pct=config.get("max_daily_loss_pct", 0.02),
            max_drawdown_pct=config.get("max_drawdown_pct", 0.05),
            max_trades_per_day=config.get("max_trades_per_day", 5),
            max_position_size=config.get("max_position_size", 2),
            max_notional_exposure=config.get("max_notional_exposure", 50_000),
            circuit_breaker_after_loss=config.get("circuit_breaker_after_loss", 3),
        )
        self.initial_capital = initial_capital
        self.state = RiskState(
            daily_start_equity=initial_capital,
            peak_equity=initial_capital,
            todays_date=date.today(),
        )

    def check_pre_trade(
        self, potential_loss: float
    ) -> tuple[bool, list[str]]:
        """Check if a trade is allowed.

        Returns:
            (allowed, list_of_reasons_if_blocked)
        """
        reasons: list[str] = []

        if self.state.daily_pnl <= -self.guardrails.max_daily_loss_usd:
            reasons.append("Daily loss limit reached")

        daily_pnl_pct = (
            self.state.daily_pnl / self.initial_capital
            if self.initial_capital > 0
            else 0
        )
        if daily_pnl_pct <= -self.guardrails.max_daily_loss_pct:
            reasons.append("Daily loss percentage limit reached")

        if self.state.max_drawdown >= self.guardrails.max_drawdown_pct:
            reasons.append("Maximum drawdown limit reached")

        if self.state.daily_trades >= self.guardrails.max_trades_per_day:
            reasons.append("Max trades per day reached")

        if potential_loss > self.guardrails.max_notional_exposure:
            reasons.append("Exposure exceeds limit")

        return len(reasons) == 0, reasons

    def record_trade(self, pnl: float, is_loss: bool) -> None:
        """Record a trade outcome and update risk state."""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1

        if is_loss:
            self.state.daily_losses += 1
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        current_equity = self.state.daily_start_equity + self.state.daily_pnl
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
            if drawdown > self.state.max_drawdown:
                self.state.max_drawdown = drawdown

    def should_stop_trading(self) -> Optional[str]:
        """Check if trading should be stopped entirely.

        Returns:
            Reason string if stopped, None if trading is allowed.
        """
        if self.state.daily_pnl <= -self.guardrails.max_daily_loss_usd:
            return (
                f"Daily loss limit ({self.guardrails.max_daily_loss_usd}) reached"
            )

        if self.state.consecutive_losses >= self.guardrails.circuit_breaker_after_loss:
            return (
                f"Circuit breaker: {self.state.consecutive_losses} consecutive losses"
            )

        if self.state.max_drawdown >= self.guardrails.max_drawdown_pct:
            return (
                f"Max drawdown ({self.guardrails.max_drawdown_pct * 100:.1f}%) reached"
            )

        return None

    def reset_daily(self) -> None:
        """Reset daily counters at start of new session."""
        self.state.daily_pnl = 0.0
        self.state.daily_trades = 0
        self.state.daily_losses = 0
        self.state.consecutive_losses = 0
        self.state.daily_start_equity = self.state.peak_equity
        self.state.todays_date = date.today()
