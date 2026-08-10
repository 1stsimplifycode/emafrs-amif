"""Speaker credibility scoring -- PDF Claim 19: C_s = f(authority, credibility, historical
impact).

HONESTY NOTE: the "historical impact" component requires a backtest correlating a source's
past communications with subsequent market movements, which does not exist yet (needs the
full pipeline + accumulated history -- see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md).
Until that exists, credibility is computed from the authority-tier component ONLY, and
`historical_component=None` is left visible on the result rather than silently defaulted to
some placeholder number, so nothing downstream can mistake an unset component for a measured one.
"""
from __future__ import annotations

from dataclasses import dataclass

# Authority tier is a documented, deliberately coarse prior on institutional weight -- not a
# measured quantity. It reflects role, nothing else (see roles.py's no-name-inference rule).
ROLE_AUTHORITY_TIER = {
    "government_policymaker": 1.0,
    "central_bank_official": 1.0,
    "corporate_executive": 0.7,
    "institutional_investor_analyst": 0.6,
    "social_media_participant": 0.3,
    "unclassified": 0.4,
}


@dataclass
class CredibilityScore:
    value: float
    authority_component: float
    historical_component: float | None
    role: str


def compute_credibility(role: str, historical_impact_corr: float | None = None) -> CredibilityScore:
    authority = ROLE_AUTHORITY_TIER.get(role, ROLE_AUTHORITY_TIER["unclassified"])
    if historical_impact_corr is None:
        return CredibilityScore(value=authority, authority_component=authority, historical_component=None, role=role)

    hist = max(0.0, min(1.0, historical_impact_corr))
    value = 0.5 * authority + 0.5 * hist
    return CredibilityScore(value=value, authority_component=authority, historical_component=hist, role=role)
