# Data Sources and APIs

All data acquisition lives in `src/emafrs/data/`. Every client returns
`(payload, SourceRecord)` — see `src/emafrs/data/schema.py` — so provenance (provider,
endpoint, fetch time, observation time, license note, cache status) travels with every
record. A client whose key is absent raises `RuntimeError` at construction; nothing in this
codebase substitutes synthetic data for a missing/unreachable source. Verified live against
real endpoints on 2026-08-10 (yfinance succeeded against live Yahoo Finance data; SEC EDGAR
hit a live 403 rate-limit/bot-defense response, which the resilience layer correctly
surfaced as `ProviderUnavailableError` rather than masking it — see `docs/DATA_LEAKAGE_AUDIT.md`
and the Resilience section below for what "correctly" means here).

## Summary table

| Provider | Purpose | Auth | Endpoint(s) used | Key fields | Free-tier limits | License / usage notes | Fallback |
|---|---|---|---|---|---|---|---|
| **yfinance** | Primary OHLCV, returns, technicals | None | Yahoo Finance chart API (via `yfinance` package) | date, OHLC, adj close, volume | No formal SLA/limit; unofficial scrape of Yahoo endpoints | Free for research/personal use; no redistribution guarantee | Alpha Vantage |
| **Alpha Vantage** | Validation/fallback OHLCV, technical indicators, intraday bars | API key | `TIME_SERIES_DAILY` (not `_ADJUSTED` — see note below), `TIME_SERIES_INTRADAY`, indicator functions (`RSI`,`MACD`,`SMA`,`EMA`,`BBANDS`) | OHLCV (not dividend-adjusted), indicator series | 25 requests/day, 5/minute (published tier as of writing — verify current limits) | Redistribution restrictions per alphavantage.co/terms_of_service | yfinance |
| **Finnhub** | Real-time quote snapshot, company news | API key | `/quote`, `/company-news` | last price, timestamp; headline/summary/source/url/datetime | 60 calls/minute | finnhub.io/terms-of-service | NewsAPI (for news) |
| **NewsAPI.org** | General market/company news | API key | `/v2/everything` | title, description, source, publishedAt, url | 100 requests/day, Developer tier, ~1 month lookback, non-commercial, attribution required | newsapi.org/terms | Finnhub company-news |
| **YouTube Data API v3** | Financial-communication video discovery + transcript text | API key | `search.list`, transcript via `youtube-transcript-api` | title, channel, publishedAt, description, transcript text | 10,000 quota units/day; `search.list`=100 units, `videos.list`=1 unit | YouTube ToS; transcripts are auto/creator captions, not verified | none (unique modality) |
| **FRED** | Macroeconomic context series | API key | `/fred/series/observations` | date, value | ~120 req/min (generous) | Public domain (U.S. government work) | World Bank/IMF (not yet implemented) |
| **SEC EDGAR** | Company filings, full-text search, CIK lookup | None (custom User-Agent required) | `company_tickers.json`, `submissions/CIK*.json`, `efts.sec.gov/LATEST/search-index` | filing form, filed date, accession no., filer names | <=10 req/sec; strict bot-defense on generic headers/User-Agents | Public domain (U.S. government work) | none |

## Verified with real keys (update)

Alpha Vantage and YouTube keys were added to `.env` and exercised for real; NewsAPI's
key was also added but does not work (see below). Findings, each also cross-referenced
in `docs/LIMITATIONS.md`:

- **Alpha Vantage**: `TIME_SERIES_DAILY_ADJUSTED` now returns `"This is a premium
  endpoint"` on the free tier — a provider policy change since this client was first
  written. Fixed by switching to the still-free `TIME_SERIES_DAILY` (no dividend-adjusted
  close column). Verified with a real live call against `AAPL`.
- **YouTube Data API v3**: works correctly for both `search.list` (found real, current
  videos, e.g. an actual "FOMC Press Conference, July 29, 2026") and transcript
  retrieval — but `youtube-transcript-api`'s underlying (unofficial) transcript endpoint
  IP-blocks a large fraction of requests from this development environment (23 of 35 in
  one real run — see `docs/TEXT_MODALITY_PILOT.md`). This is a provider-side anti-bot
  heuristic, not a defect in this project's code, and is not fully mitigated by the
  courtesy delay this project added between requests.
- **NewsAPI.org**: the specific key provided for this project is rejected directly by
  newsapi.org (`401 apiKeyInvalid`), verified with a raw HTTP request bypassing all
  project code. `CONFIG.is_configured("newsapi")` correctly reports the key as *present*
  (it is syntactically a key), but live calls fail until a valid key is supplied.

## Resilience behavior (`src/emafrs/data/base.py`, `cache.py`)

1. Cache-through GET with source-specific TTL (market data: minutes; news: 30 min; macro:
   24h; SEC company tickers: 7 days).
2. Exponential backoff with jitter (4 attempts) on timeouts, connection errors, and 5xx.
3. On 429 or persistent failure: fall back to the **last cached response regardless of
   TTL**, but the returned `SourceRecord.cache_hit=True` and `fetched_at` reflects the
   original (stale) fetch time — callers/dashboard must render this as stale, never as live.
4. If no cache exists at all (first-ever call during an outage), the client raises
   `ProviderUnavailableError` rather than returning empty/zero/fabricated data. This was
   observed directly during integration testing: a first-ever SEC EDGAR call hit a live
   403 during development, and the client correctly propagated the failure instead of
   masking it.

## YouTube modality honesty

The YouTube client (`youtube_client.py`) only ever produces **video metadata** and
**transcript text**. It does not download or process audio/video streams. Every
transcript-derived record is tagged `modality="text_derived_from_av_source"`; the ASEM and
any dashboard surfacing it must report this as *"Textual modality derived from audiovisual
communication source,"* never as audio or video modeling, per the build brief's explicit
instruction against overclaiming modality coverage. True audio/video affect features (pitch,
energy, facial expression) are only attempted for a small hand-picked case-study set (see
`docs/LIMITATIONS.md`), not the live pipeline.

## What is explicitly NOT implemented

- World Bank / IMF / GDELT / Stooq — not wired in. Three reliable, provenance-tracked
  sources are worth more than a longer list of thin integrations (per build brief Section
  12: "data quality > API quantity"). Can be added the same way if a specific experiment
  needs them.
- True Level-2/Level-3 tick order book data — no free/research-tier provider grants this
  for equities. See `docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md` row #4 for the documented
  proxy approach used instead.

## Configuration

Copy `.env.example` to `.env` and fill in whichever keys you have:

```
YOUTUBE_API_KEY=
NEWS_API_KEY=
ALPHAVANTAGE_API_KEY=
FINNHUB_API_KEY=
FRED_API_KEY=
SEC_EDGAR_USER_AGENT=YourName your.email@example.com
```

`emafrs.data.config.CONFIG.configured_sources()` reports which sources are actually usable
at runtime; `emafrs.data.health.check_all(live=True)` probes each configured source with one
real (cheap) call — this is run sparingly, not on every dashboard refresh, because the
YouTube probe alone costs 100 quota units.
