"""Citizen/stakeholder alert generation -- PDF Section 6.9, Fig. 4, Claim 16.

Turns a manipulation risk score + explanation into a structured, asynchronously-
deliverable alert. Delivery channels (push notification, email, webhook) are NOT
implemented -- no real delivery infrastructure (APNs/FCM credentials, an email provider)
is configured in this reconstruction, so this module produces the alert PAYLOAD only and
documents delivery as not yet built, rather than faking a "sent" status.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class CitizenAlert:
    ticker: str
    triggered_at: datetime
    risk_score: float
    threshold: float
    headline: str
    explanation_text: str
    suspicious_asset: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "triggered_at": self.triggered_at.isoformat(),
            "risk_score": self.risk_score,
            "threshold": self.threshold,
            "headline": self.headline,
            "explanation_text": self.explanation_text,
            "suspicious_asset": self.suspicious_asset,
        }


def maybe_generate_alert(
    ticker: str, risk_score: float, threshold: float, explanation_text: str
) -> CitizenAlert | None:
    """Returns a CitizenAlert only if risk_score exceeds threshold, else None -- callers
    must not alert on every prediction, only genuinely elevated ones (PDF: "When Risk_t >
    tau, the system flags possible pump-and-dump activity")."""
    if risk_score <= threshold:
        return None

    headline = (
        f"Elevated manipulation risk detected for {ticker.upper()} "
        f"(score {risk_score:.2f}, threshold {threshold:.2f}). "
        "This is a model-computed risk score, not a confirmed finding of manipulation."
    )
    return CitizenAlert(
        ticker=ticker.upper(),
        triggered_at=datetime.now(timezone.utc),
        risk_score=risk_score,
        threshold=threshold,
        headline=headline,
        explanation_text=explanation_text,
        suspicious_asset=ticker.upper(),
    )
