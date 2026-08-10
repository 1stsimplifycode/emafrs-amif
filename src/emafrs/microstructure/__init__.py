"""Market Microstructure Intelligence Engine (MMIE) -- PDF Section 6.6.

Every signal here is a documented PROXY derived from OHLCV bars, not true order-book data.
See emafrs.microstructure.proxies module docstring and docs/LIMITATIONS.md.
"""
from emafrs.microstructure.proxies import build_microstructure_features

__all__ = ["build_microstructure_features"]
