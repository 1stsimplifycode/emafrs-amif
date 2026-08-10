"""Citizen/stakeholder alert generation -- PDF Section 6.9. Payload generation only;
delivery channels (push/email/webhook) are not yet built -- see docs/MLOPS.md."""
from emafrs.alerts.citizen_alerts import CitizenAlert, maybe_generate_alert

__all__ = ["CitizenAlert", "maybe_generate_alert"]
