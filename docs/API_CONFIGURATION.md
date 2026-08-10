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

As of the initial research phases (data acquisition through experiment protocol), only
yfinance and SEC EDGAR had been exercised against live data. YouTube, Alpha Vantage, and
NewsAPI keys were added afterward and are now verified working (see
`docs/DATA_SOURCES_AND_APIS.md`'s "Verified with real keys" section for the two real
compatibility bugs and one key-format issue found and fixed along the way). Finnhub and
FRED keys are still not configured. See `docs/LIMITATIONS.md` and
`docs/TEXT_MODALITY_PILOT.md` for what real YouTube-derived text data has (and hasn't
yet) unlocked for the text/affective modality.

## Rate limit and caching behavior

All keyed clients cache-through a local TTL disk cache (`data/cache/`, gitignored) with
provider-appropriate TTLs and a stale-cache fallback on rate-limit/outage — see
`src/emafrs/data/base.py` and `docs/DATA_SOURCES_AND_APIS.md`'s "Resilience behavior"
section. This means repeated calls during development do not burn through daily quotas as
fast as the raw per-provider limits might suggest, but it also means a stale cache can be
served during an outage — `SourceRecord.cache_hit` and `fetched_at` always reveal this
distinction if a consumer needs to know.
