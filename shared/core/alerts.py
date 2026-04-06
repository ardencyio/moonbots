"""Alert management for trading bots."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import requests


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    bot_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


class AlertManager:
    """Manages alerts with optional webhook notifications."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.alerts: list[Alert] = []

    def raise_alert(self, alert: Alert) -> None:
        self.alerts.append(alert)
        if alert.severity == AlertSeverity.CRITICAL and self.webhook_url:
            self._send_webhook(alert)

    def _send_webhook(self, alert: Alert) -> None:
        payload = {
            "content": f"**[{alert.severity.value.upper()}] {alert.bot_id}** — {alert.title}: {alert.message}"
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=10)
        except Exception:
            pass  # Never break trading logic on alert failure
