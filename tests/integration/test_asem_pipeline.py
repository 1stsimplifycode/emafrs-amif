"""Integration tests for the ASEM sub-models: these download and run real transformer
models (CPU), so they're slower and marked `integration`. No network/model call is ever
mocked here -- the point is to prove the real pipeline behaves sensibly on real text.
"""
import pytest

pytestmark = pytest.mark.integration


def test_emotion_detector_distinguishes_fear_from_confidence():
    from emafrs.nlp.emotion import ZeroShotEmotionDetector

    detector = ZeroShotEmotionDetector()
    fearful_text = "Investors are panicking as the company faces imminent bankruptcy and total collapse."
    confident_text = "Our balance sheet is exceptionally strong and we are fully confident in our growth trajectory."

    fear_scores = detector.score(fearful_text)
    confident_scores = detector.score(confident_text)

    assert fear_scores.scores["fear"] > confident_scores.scores["fear"]
    assert confident_scores.scores["confidence"] > fear_scores.scores["confidence"]


def test_semantic_encoder_similar_sentences_closer_than_unrelated():
    import numpy as np

    from emafrs.nlp.semantic_encoder import ContextualSemanticEncoder

    encoder = ContextualSemanticEncoder()
    a = encoder.encode_one("The Federal Reserve raised interest rates by 25 basis points.")
    b = encoder.encode_one("The central bank increased its policy rate by a quarter point.")
    c = encoder.encode_one("The chef added a pinch of salt to the simmering tomato sauce.")

    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    assert sim_ab > sim_ac


def test_sentiment_classifier_distinguishes_polarity():
    from emafrs.nlp.sentiment import SentimentPolarityClassifier

    clf = SentimentPolarityClassifier()
    positive = clf.score("The company reported record profits and raised its full-year guidance.")
    negative = clf.score("The company reported a massive loss and slashed its dividend.")

    assert positive.polarity > negative.polarity


def test_asem_end_to_end_produces_consistent_feature_vector():
    from emafrs.nlp.asem import AffectiveSignalExtractionModule

    asem = AffectiveSignalExtractionModule()
    result = asem.process(
        text="The Federal Reserve chair expressed strong confidence in the economy's resilience.",
        provider="youtube",
        entity="FOMC",
        metadata={"channel_id": "unverified_channel"},
    )

    assert result.role == "unclassified"  # channel not in the verified allowlist
    assert 0.0 <= result.hype_score <= 1.0
    assert result.semantic_embedding.shape[0] > 0
    vec = result.as_vector()
    expected_len = 7 + 1 + 1 + result.semantic_embedding.shape[0] + 6  # emotion+sent+intensity+embed+role
    assert vec.shape[0] == expected_len
