"""Influencer role stratification (PDF Section 6.4.1) -- 5 role categories.

Rule (build brief Section 18/19): "Do not infer someone's institutional role without
evidence." Role is assigned ONLY from structural/verified metadata (which provider a
communication came from, its filing type, or a manually-verified channel ID), never from a
speaker's name or the text content. The default for everything else is "unclassified" --
expected to be the common case with the DAL sources currently wired in (NewsAPI/Finnhub
article `source.name` fields do not reliably indicate the role of a quoted speaker, so
neither is used for role inference).
"""
from __future__ import annotations

# PDF Eq. (1) defines rho_t as a one-hot over Nr=5 role categories. This reconstruction
# adds a 6th "unclassified" bucket rather than forcing every real-world source into one of
# the 5 speculative categories when the evidence doesn't support it -- a disclosed,
# deliberate deviation (see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md), not a silent one.
ROLE_CATEGORIES = [
    "government_policymaker",
    "central_bank_official",
    "corporate_executive",
    "institutional_investor_analyst",
    "social_media_participant",
    "unclassified",
]

# Populate only with identifiers actually verified as belonging to the named institution.
# Empty by default -- this is a deliberate allowlist, not a heuristic guess, and growing it
# is a manual, documented act, never an inference from a channel/display name string.
VERIFIED_YOUTUBE_CHANNELS: dict[str, str] = {
    # "UC_verified_channel_id_here": "central_bank_official",
}

# SEC EDGAR filings are, by the structure of the filing system itself (not by inferring
# anything about a named individual), official disclosures made on behalf of a company --
# this is a structural fact about the source, not a guess about a person's role.
_PROVIDER_STRUCTURAL_ROLE = {
    "sec_edgar": "corporate_executive",
}


def infer_role(provider: str, metadata: dict | None = None) -> str:
    metadata = metadata or {}
    if provider in _PROVIDER_STRUCTURAL_ROLE:
        return _PROVIDER_STRUCTURAL_ROLE[provider]
    if provider == "youtube":
        channel_id = metadata.get("channel_id")
        if channel_id and channel_id in VERIFIED_YOUTUBE_CHANNELS:
            return VERIFIED_YOUTUBE_CHANNELS[channel_id]
        return "unclassified"
    return "unclassified"


def role_to_onehot(role: str) -> list[int]:
    if role not in ROLE_CATEGORIES:
        role = "unclassified"
    return [1 if r == role else 0 for r in ROLE_CATEGORIES]
