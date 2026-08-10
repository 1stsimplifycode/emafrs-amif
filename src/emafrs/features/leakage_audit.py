"""Automated data-leakage checks (build brief Section 25: "mandatory").

Three independent, mechanically-checkable properties are audited here, not just asserted
in prose:

1. No-lookahead in rolling/windowed features: truncating history at t and recomputing must
   give the same value at t as computing on the full series. A feature function that peeks
   ahead will fail this for at least one sampled t.
2. No future-news leakage: every news/communication record used to build a feature for
   prediction time T must have an observation timestamp <= T.
3. Chronological (non-shuffled) train/test splits: every test timestamp must be later than
   every train timestamp.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class LeakageAuditResult:
    check_name: str
    passed: bool
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def assert_no_lookahead(
    feature_fn: Callable[[pd.DataFrame], pd.Series],
    df: pd.DataFrame,
    n_samples: int = 8,
    min_index: int = 60,
    seed: int = 0,
) -> LeakageAuditResult:
    """For several cutoff points t, recomputes feature_fn on df.iloc[:t+1] and compares the
    last value to feature_fn(df).iloc[t]. A mismatch means feature_fn used rows beyond t.

    `min_index` skips the initial warm-up region where most rolling features are legitimately
    NaN on both sides (not informative for this check).
    """
    n = len(df)
    if n <= min_index + 1:
        return LeakageAuditResult("no_lookahead", passed=False, details={"error": "series too short"})

    full = feature_fn(df)
    rng = random.Random(seed)
    candidates = list(range(min_index, n))
    sample = rng.sample(candidates, min(n_samples, len(candidates)))

    mismatches = []
    for t in sample:
        truncated = feature_fn(df.iloc[: t + 1])
        full_val = full.iloc[t]
        trunc_val = truncated.iloc[-1]
        both_nan = pd.isna(full_val) and pd.isna(trunc_val)
        close_enough = both_nan or (
            pd.notna(full_val) and pd.notna(trunc_val) and np.isclose(full_val, trunc_val, rtol=1e-8, atol=1e-10)
        )
        if not close_enough:
            mismatches.append({"t": int(t), "full_val": full_val, "truncated_val": trunc_val})

    return LeakageAuditResult(
        "no_lookahead",
        passed=len(mismatches) == 0,
        details={"n_sampled": len(sample), "mismatches": mismatches},
    )


def check_news_before_prediction_time(
    records: list[dict],
    prediction_time: datetime,
    timestamp_key: str = "publishedAt",
) -> LeakageAuditResult:
    """Every record's timestamp must be <= prediction_time. Records with unparseable
    timestamps are treated as violations (fail closed, not open)."""
    violations = []
    for r in records:
        raw_ts = r.get(timestamp_key)
        try:
            ts = raw_ts if isinstance(raw_ts, datetime) else datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            violations.append({"record": r, "reason": "unparseable timestamp"})
            continue
        pt = prediction_time
        # normalize tz-awareness for comparison
        if ts.tzinfo is None and pt.tzinfo is not None:
            pt = pt.replace(tzinfo=None)
        elif ts.tzinfo is not None and pt.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        if ts > pt:
            violations.append({"record_timestamp": ts.isoformat(), "prediction_time": pt.isoformat()})

    return LeakageAuditResult(
        "news_before_prediction_time",
        passed=len(violations) == 0,
        details={"n_records": len(records), "violations": violations},
    )


def check_chronological_split(
    train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex
) -> LeakageAuditResult:
    """Every test date must be strictly later than every train date (no shuffled/random
    k-fold split across time, which would leak future structure into training)."""
    if len(train_dates) == 0 or len(test_dates) == 0:
        return LeakageAuditResult("chronological_split", passed=False, details={"error": "empty split"})
    max_train = max(train_dates)
    min_test = min(test_dates)
    passed = max_train < min_test
    overlap = sorted(set(train_dates) & set(test_dates))
    return LeakageAuditResult(
        "chronological_split",
        passed=passed and not overlap,
        details={
            "max_train_date": str(max_train),
            "min_test_date": str(min_test),
            "n_overlapping_dates": len(overlap),
        },
    )


def check_rolling_zscore_uses_only_past(series: pd.Series, window: int = 63, n_samples: int = 8) -> LeakageAuditResult:
    from emafrs.features.preprocessing import rolling_zscore

    df = series.to_frame("x")
    return assert_no_lookahead(lambda d: rolling_zscore(d["x"], window=window), df, n_samples=n_samples)
