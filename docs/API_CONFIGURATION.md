# API Configuration

Full source-by-source details (endpoints, fields, rate limits, licensing) are in
`docs/DATA_SOURCES_AND_APIS.md`. This document is the short, ops-focused setup guide.

## Setup

```bash
cp .env.example .env
```

Then fill in whichever keys you have. Every key is optional — `emafrs.data.config.CONFIG`
detects which sources are usable and every DAL client raises a clear `RuntimeError` at
construction time if its key is missing, rather than silently degrading to fake data.

| Variable | Required for | Where to get it |
|---|---|---|
| `YOUTUBE_API_KEY` | YouTube video/transcript ingestion | console.cloud.google.com (enable "YouTube Data API v3") |
| `NEWS_API_KEY` | General news search | newsapi.org/register |
| `ALPHAVANTAGE_API_KEY` | Validation market data, intraday bars | alphavantage.co/support/#api-key |
| `FINNHUB_API_KEY` | Real-time quotes, company news | finnhub.io/register |
| `FRED_API_KEY` | Macroeconomic series | fred.stlouisfed.org/docs/api/api_key.html |
| `SEC_EDGAR_USER_AGENT` | SEC EDGAR (no key, but required header) | any descriptive string with real contact info, e.g. `"YourName your.email@example.com"` |

`yfinance` and `SEC EDGAR` need no key (though EDGAR needs the User-Agent above per its
fair-access policy).

## Checking what's configured

```python
from emafrs.data.config import CONFIG
print(CONFIG.configured_sources())
```

or via the helper MCP server's `list_configured_sources` tool, or
`emafrs.data.health.check_all(live=True)` for an actual reachability probe (costs real
quota on some providers — notably YouTube's `search.list` at 100 units/call out of a
10,000/day budget; don't call this on every dashboard refresh).

## Never commit `.env`

`.gitignore` already excludes it. If you ever see `.env` in `git status`, do not commit
it — check `git diff --cached` before pushing if unsure.

## What is currently configured in this development environment

As of the research phases (data acquisition through experiment protocol), **only
yfinance and SEC EDGAR were exercised against live data** — no NewsAPI/YouTube/Alpha
Vantage/Finnhub/FRED key was configured. See `docs/LIMITATIONS.md` for what this means
for the text/affective modality (not populated in any trained dataset yet) and
`docs/DATA_SOURCES_AND_APIS.md` for the one live SEC EDGAR rate-limit event observed
during DAL development (correctly surfaced as an error, not masked).

## Rate limit and caching behavior

All keyed clients cache-through a local TTL disk cache (`data/cache/`, gitignored) with
provider-appropriate TTLs and a stale-cache fallback on rate-limit/outage — see
`src/emafrs/data/base.py` and `docs/DATA_SOURCES_AND_APIS.md`'s "Resilience behavior"
section. This means repeated calls during development do not burn through daily quotas as
fast as the raw per-provider limits might suggest, but it also means a stale cache can be
served during an outage — `SourceRecord.cache_hit` and `fetched_at` always reveal this
distinction if a consumer needs to know.
