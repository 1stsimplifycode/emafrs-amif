"""Persistence layer: SQLAlchemy models + session management. SQLite by default
(DATABASE_URL env var to point elsewhere, e.g. Postgres in production)."""
from emafrs.db.session import get_session, init_db

__all__ = ["get_session", "init_db"]
