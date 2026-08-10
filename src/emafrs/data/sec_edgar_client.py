"""SEC EDGAR client: company filings and full-text search. No API key; requires a
descriptive User-Agent per SEC's fair-access policy (see .env.example SEC_EDGAR_USER_AGENT).

Rate limit: SEC requests <=10 requests/second; this client stays well under that via the
shared retry/backoff and cache-through layer. See sec.gov/os/accessing-edgar-data.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from emafrs.data.base import BaseAPIClient
from emafrs.data.config import CONFIG
from emafrs.data.schema import SourceRecord

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

LICENSE_NOTE = "SEC EDGAR data is a U.S. government work (public domain). See sec.gov/privacy."


class SECEdgarClient(BaseAPIClient):
    provider = "sec_edgar"

    def __init__(self):
        # SEC's automated-tooling guidance (sec.gov/os/webmaster-faq#developers) asks for a
        # descriptive User-Agent with contact info; some responses also require an explicit
        # Accept-Encoding/Host-friendly header set to avoid being classified as a bot.
        self.headers = {
            "User-Agent": CONFIG.sec_edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html",
        }

    def _headers_for(self, endpoint: str) -> dict:
        return self.headers

    def ticker_to_cik(self, ticker: str) -> tuple[str | None, SourceRecord]:
        payload, fetched_at, cache_hit = self.cached_get_json(
            "company_tickers", _TICKERS_URL, headers=self.headers, ttl_seconds=7 * 24 * 3600
        )
        cik = None
        for row in payload.values():
            if row.get("ticker", "").upper() == ticker.upper():
                cik = str(row["cik_str"]).zfill(10)
                break
        record = self.make_record(
            "company_tickers", fields={"ticker": ticker, "found": cik is not None},
            fetched_at=fetched_at, entity=ticker, license_note=LICENSE_NOTE, cache_hit=cache_hit,
        )
        return cik, record

    def get_submissions(self, cik10: str) -> tuple[dict, SourceRecord]:
        url = _SUBMISSIONS_URL.format(cik10=cik10)
        payload, fetched_at, cache_hit = self.cached_get_json(
            "submissions", url, headers=self.headers, ttl_seconds=6 * 3600
        )
        recent = payload.get("filings", {}).get("recent", {})
        n_filings = len(recent.get("form", []))
        record = self.make_record(
            "submissions", fields={"cik": cik10, "n_recent_filings": n_filings},
            fetched_at=fetched_at, entity=cik10, license_note=LICENSE_NOTE, cache_hit=cache_hit,
        )
        return payload, record

    def full_text_search(
        self, query: str, forms: str | None = None, start: date | None = None, end: date | None = None
    ) -> tuple[list[dict], SourceRecord]:
        """Full-text search over filings since 2001. `forms` e.g. '8-K' to restrict to
        material-event disclosures relevant to event/hype detection."""
        params: dict = {"q": query}
        if forms:
            params["forms"] = forms
        if start:
            params["startdt"] = start.isoformat()
        if end:
            params["enddt"] = end.isoformat()

        payload, fetched_at, cache_hit = self.cached_get_json(
            "fulltext_search", _FULLTEXT_SEARCH_URL, params=params, headers=self.headers, ttl_seconds=3600
        )
        hits = payload.get("hits", {}).get("hits", [])
        results = [
            {
                "accession_no": h["_id"],
                "entity": h["_source"].get("display_names"),
                "form": h["_source"].get("forms"),
                "filed_at": h["_source"].get("file_date"),
            }
            for h in hits
        ]
        record = self.make_record(
            "fulltext_search", fields={"query": query, "n_results": len(results), "forms": forms},
            fetched_at=fetched_at, entity=query, license_note=LICENSE_NOTE, cache_hit=cache_hit,
        )
        return results, record
