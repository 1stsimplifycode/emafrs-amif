from emafrs.alerts.citizen_alerts import maybe_generate_alert


def test_no_alert_below_threshold():
    alert = maybe_generate_alert("AAPL", risk_score=0.3, threshold=0.7, explanation_text="calm")
    assert alert is None


def test_alert_generated_above_threshold():
    alert = maybe_generate_alert("AAPL", risk_score=0.85, threshold=0.7, explanation_text="explanation here")
    assert alert is not None
    assert alert.ticker == "AAPL"
    assert alert.risk_score == 0.85
    assert "0.85" in alert.headline
    assert "not a confirmed finding" in alert.headline
    assert alert.explanation_text == "explanation here"


def test_ticker_normalized_to_uppercase():
    alert = maybe_generate_alert("aapl", risk_score=0.9, threshold=0.5, explanation_text="x")
    assert alert.ticker == "AAPL"
    assert alert.suspicious_asset == "AAPL"


def test_to_dict_roundtrip_shape():
    alert = maybe_generate_alert("MSFT", risk_score=0.95, threshold=0.6, explanation_text="x")
    d = alert.to_dict()
    assert set(d.keys()) == {
        "ticker", "triggered_at", "risk_score", "threshold", "headline", "explanation_text", "suspicious_asset",
    }


def test_exactly_at_threshold_does_not_alert():
    """Strict inequality: risk_score == threshold must not trigger, matching PDF's
    "Risk_t > tau" (not >=)."""
    alert = maybe_generate_alert("AAPL", risk_score=0.5, threshold=0.5, explanation_text="x")
    assert alert is None
