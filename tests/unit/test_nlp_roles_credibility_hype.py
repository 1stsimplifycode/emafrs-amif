"""Unit tests for the parts of ASEM that don't require loading a transformer model:
role inference, credibility scoring, hype score assembly."""
from emafrs.nlp.credibility import ROLE_AUTHORITY_TIER, compute_credibility
from emafrs.nlp.emotion import EMOTION_CATEGORIES, EmotionScores
from emafrs.nlp.hype import compute_hype_score, hype_acceleration, spike_alert, text_hype_component
from emafrs.nlp.roles import ROLE_CATEGORIES, infer_role, role_to_onehot


def test_infer_role_never_uses_name_only_structural_metadata():
    assert infer_role("sec_edgar", {}) == "corporate_executive"
    assert infer_role("newsapi", {"source": {"name": "Reuters"}}) == "unclassified"
    assert infer_role("youtube", {"channel_id": "not_in_allowlist"}) == "unclassified"


def test_role_to_onehot_shape_and_correctness():
    onehot = role_to_onehot("central_bank_official")
    assert len(onehot) == len(ROLE_CATEGORIES)
    assert sum(onehot) == 1
    assert onehot[ROLE_CATEGORIES.index("central_bank_official")] == 1


def test_role_to_onehot_falls_back_to_unclassified_for_unknown_string():
    onehot = role_to_onehot("not_a_real_role")
    assert onehot[ROLE_CATEGORIES.index("unclassified")] == 1


def test_credibility_without_historical_data_uses_authority_only():
    result = compute_credibility("government_policymaker")
    assert result.historical_component is None
    assert result.value == ROLE_AUTHORITY_TIER["government_policymaker"]


def test_credibility_with_historical_data_blends_both_components():
    result = compute_credibility("social_media_participant", historical_impact_corr=0.9)
    assert result.historical_component == 0.9
    expected = 0.5 * ROLE_AUTHORITY_TIER["social_media_participant"] + 0.5 * 0.9
    assert abs(result.value - expected) < 1e-9


def test_credibility_clips_historical_component_to_unit_interval():
    result = compute_credibility("unclassified", historical_impact_corr=5.0)
    assert result.historical_component == 1.0


def _make_emotion_scores(**overrides) -> EmotionScores:
    base = {c: 0.0 for c in EMOTION_CATEGORIES}
    base.update(overrides)
    return EmotionScores(scores=base)


def test_text_hype_component_high_for_optimistic_assertive_confident_language():
    hype_emotion = _make_emotion_scores(optimism=0.9, assertiveness=0.9, confidence=0.9)
    calm_emotion = _make_emotion_scores(caution=0.9, uncertainty=0.9, evasiveness=0.9)
    assert text_hype_component(hype_emotion) > text_hype_component(calm_emotion)


def test_compute_hype_score_scales_with_credibility():
    emotion = _make_emotion_scores(optimism=0.9, assertiveness=0.9, confidence=0.9)
    low_cred_score = compute_hype_score(credibility=0.2, emotion_scores=emotion)
    high_cred_score = compute_hype_score(credibility=0.9, emotion_scores=emotion)
    assert high_cred_score > low_cred_score


def test_hype_acceleration_and_spike_alert():
    import pandas as pd

    hype_series = pd.Series([0.1, 0.15, 0.9, 0.92])
    accel = hype_acceleration(hype_series)
    assert accel.iloc[0] != accel.iloc[0]  # NaN for first element (no prior value)
    spikes = spike_alert(accel, threshold=0.5)
    assert spikes.iloc[2] == True  # noqa: E712 -- large jump from 0.15 -> 0.9
    assert spikes.iloc[1] == False  # noqa: E712
