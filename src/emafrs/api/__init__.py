"""FastAPI backend -- build brief Section 28. Run with:

    uvicorn emafrs.api.app:app --reload
"""
from emafrs.api.app import app

__all__ = ["app"]
