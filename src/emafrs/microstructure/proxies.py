"""Market Microstructure Intelligence Engine (MMIE) -- PDF Section 6.6.

HONESTY NOTE (see docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md row #4): no free/research-tier
data source used in this reconstruction provides true Level-2/Level-3 order book depth,
bid/ask quotes, or trade-by-trade tick data. Every function in this module computes a
DOCUMENTED PROXY derived from OHLCV bars (yfinance/Alpha Vantage intraday), never real
order-book measurements. Every output name/column is suffixed `_proxy` and every docstring
states the specific approximation and what real signal it stands in for. Where a signal has
NO defensible bar-level proxy (true per-price-level depth depletion requires actual L2 quote
data), this module returns NaN / raises rather than fabricating a number that looks real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def obi_proxy(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Order Book Imbalance PROXY, via the Close Location Value (CLV) -- a standard
    technical-analysis technique (the buy/sell volume split used in the Accumulation/
    Distribution Line) repurposed here as a directional-pressure proxy for PDF's OBI_t:

        obi_proxy_t = (2*close_t - high_t - low_t) / (high_t - low_t)

    Rationale: absent real bid/ask volumes, a close near the bar's high is treated as
    evidence of buy-side pressure within that bar, and a close near the low as sell-side
    pressure. Bounded in [-1, +1] like the PDF's true OBI_t, but this is NOT computed from
    actual bid/ask quoted volume and must never be presented as such.
    """
    rng = (high - low).replace(0.0, np.nan)
    return (2 * close - high - low) / rng


def delta_obi_proxy(obi: pd.Series, lag: int = 1) -> pd.Series:
    """PDF Eq.(5): rate of change of OBI. Applied here to obi_proxy, not true OBI."""
    return obi - obi.shift(lag)


def liquidity_shock_proxy(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """Liquidity Shock Indicator PROXY. True LSI (PDF Eq.6) needs the quoted bid-ask
    spread; absent quotes, this uses the intrabar high-low range as a fraction of price
    (a standard liquidity/volatility proxy: wider realized range relative to price is
    associated with wider effective spreads and thinner liquidity) and measures its
    deviation from a rolling baseline, in the same functional form as PDF Eq.6:

        range_pct_t = (high_t - low_t) / close_t
        lsi_proxy_t = |range_pct_t - mean(range_pct)| / mean(range_pct)
    """
    range_pct = (high - low) / close
    baseline = range_pct.rolling(window, min_periods=window).mean()
    return (range_pct - baseline).abs() / baseline


def ofi_proxy(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Order Flow Imbalance PROXY via the classic tick rule (Lee & Ready 1991): a bar is
    classified as buy-initiated if its close is above the prior close, sell-initiated
    otherwise, and its full volume is signed accordingly. True OFI (PDF Eq.8) requires
    actual marketable buy/sell executed volume; this is a bar-level approximation.

        ofi_proxy_t = sign(close_t - close_{t-1}) * volume_t
    """
    direction = np.sign(close.diff())
    direction = direction.replace(0.0, np.nan).ffill().fillna(0.0)  # flat bars inherit prior direction
    return direction * volume


def vpin_proxy(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    window: int = 390, n_buckets: int = 20,
) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading PROXY (PDF Eq.9). True VPIN
    requires trade-classified executions bucketed by fixed volume; this approximates the
    buy/sell split per bucket using the same CLV-based estimate as obi_proxy.

    LEAKAGE NOTE: bucket boundaries must be formed from volume observed up to and
    including the current bar only. An earlier version of this function computed bucket
    size from the WHOLE series' total volume (`cum_vol.iloc[-1] / n_buckets`), which makes
    every bucket boundary -- and therefore every output value, including at the start of
    the series -- depend on bars that haven't happened yet. That version failed
    `leakage_audit.assert_no_lookahead` during implementation and was replaced with this
    trailing-window version: at each t, only bars in [t-window+1, t] are used to both size
    and assign buckets, so vpin_proxy_t is a strict function of data at or before t.
    `window` defaults to 390 (one U.S. trading day of 1-minute bars).
    """
    idx = close.index
    high_v, low_v, close_v, volume_v = high.to_numpy(), low.to_numpy(), close.to_numpy(), volume.to_numpy(dtype=float)
    n = len(volume_v)

    rng = high_v - low_v
    with np.errstate(invalid="ignore", divide="ignore"):
        buy_est = np.where(rng != 0, volume_v * (close_v - low_v) / rng, volume_v / 2)
    sell_est = volume_v - buy_est
    imbalance = np.abs(buy_est - sell_est)

    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        start = t - window + 1
        vol_w = volume_v[start : t + 1]
        total_vol = vol_w.sum()
        if total_vol <= 0:
            continue
        bucket_size = total_vol / n_buckets
        cum_vol_w = np.cumsum(vol_w)
        bucket_id_w = np.minimum((cum_vol_w // bucket_size).astype(int), n_buckets - 1)

        imb_w = imbalance[start : t + 1]
        bucket_imb = np.zeros(n_buckets)
        bucket_vol = np.zeros(n_buckets)
        np.add.at(bucket_imb, bucket_id_w, imb_w)
        np.add.at(bucket_vol, bucket_id_w, vol_w)
        valid = bucket_vol > 0
        if valid.any():
            out[t] = np.mean(bucket_imb[valid] / bucket_vol[valid])

    return pd.Series(out, index=idx)


def depth_depletion_rate(*args, **kwargs):
    """Depth Depletion Rate (PDF Eq.7) requires true per-price-level quoted order book
    depth. There is NO defensible bar-level OHLCV proxy for this -- any number produced
    here would be fabricated, not approximated. Deliberately unimplemented; documented in
    docs/LIMITATIONS.md rather than silently omitted."""
    raise NotImplementedError(
        "depth_depletion_rate requires true Level-2 order book depth, which no configured "
        "data source provides. See docs/PDF_TO_IMPLEMENTATION_TRACEABILITY.md row #4 and "
        "docs/LIMITATIONS.md."
    )


def alpha_persistence(signal: pd.Series, forward_returns: pd.Series, window: int = 63) -> pd.Series:
    """PDF Eq.(10): rolling correlation between a generated signal and its forward
    realized return at the SAME lag the caller already applied to `forward_returns` (this
    function does not itself shift -- pass forward_returns pre-shifted by the desired
    lag, e.g. `close.pct_change().shift(-delta)`, to avoid ambiguity about lag direction)."""
    return signal.rolling(window, min_periods=window).corr(forward_returns)


def alpha_decay(alpha: pd.Series, decay_rate: float, elapsed_periods: pd.Series | float = 1.0) -> pd.Series:
    """PDF Eq.(11): alpha_decay_t = alpha_t * exp(-lambda_alpha * elapsed). `decay_rate`
    (lambda_alpha) must be supplied by the caller -- typically estimated from a rolling
    regression of realized signal decay (see alpha_persistence); not hardcoded here."""
    return alpha * np.exp(-decay_rate * elapsed_periods)


def volume_anomaly(volume: pd.Series, window: int = 21) -> pd.Series:
    """Reuses the same rolling z-score construction as features/technical.py's
    volume_zscore, exposed here under the MMIE's naming (PDF Sec 6.6.5's Vol_anomaly term)."""
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std


def build_microstructure_features(ohlcv: pd.DataFrame, vol_window: int = 21) -> pd.DataFrame:
    """Assembles all PROXY microstructure signals from an OHLCV DataFrame (columns:
    open/high/low/close/volume). Intended for intraday bars (1m/5m) where available, but
    works on any OHLCV resolution -- the coarser the bars, the noisier these proxies are,
    which is itself worth surfacing rather than hiding (see evaluation in
    docs/EXPERIMENT_PROTOCOL.md once ablations are run)."""
    high, low, close, volume = ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]

    out = pd.DataFrame(index=ohlcv.index)
    out["obi_proxy"] = obi_proxy(high, low, close)
    out["delta_obi_proxy"] = delta_obi_proxy(out["obi_proxy"])
    out["liquidity_shock_proxy"] = liquidity_shock_proxy(high, low, close, window=vol_window)
    out["ofi_proxy"] = ofi_proxy(close, volume)
    out["vpin_proxy"] = vpin_proxy(high, low, close, volume)
    out["volume_anomaly"] = volume_anomaly(volume, window=vol_window)
    return out
