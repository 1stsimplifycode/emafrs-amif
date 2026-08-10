"""Data Acquisition Layer (DAL).

Each client below wraps exactly one external provider and returns
(payload, emafrs.data.schema.SourceRecord) so every downstream consumer can see where
data came from, when it was fetched, and how fresh it is. Clients that need a key raise
RuntimeError at construction time if the key is absent -- callers must catch this and
report the source as unavailable rather than substituting synthetic data.
"""
from emafrs.data.config import CONFIG

__all__ = ["CONFIG"]
