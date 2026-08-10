"""Affective Signal Extraction Module (ASEM) and NLP transformer benchmark -- PDF Section 6.4.

emafrs.nlp.asem.AffectiveSignalExtractionModule is the production entry point, assembling
the affective feature vector a_t = [e_t || s_t || i_t || c_t || rho_t] (Eq. 1) from a single
communication. Model choices are recorded in configs/nlp_models.yaml and justified by the
benchmark in emafrs.nlp.sentiment_benchmark / docs/NLP_MODEL_BENCHMARK.md.
"""
from emafrs.nlp.asem import AffectiveFeatureVector, AffectiveSignalExtractionModule

__all__ = ["AffectiveFeatureVector", "AffectiveSignalExtractionModule"]
