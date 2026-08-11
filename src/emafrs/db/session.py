"""Database engine/session factory. Reads DATABASE_URL from the environment (see
.env.example) -- defaults to a local SQLite file for development, but the ORM layer
(models.py) uses only SQLAlchemy-portable types so pointing DATABASE_URL at Postgres in
production requires no model changes.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

DEFAULT_SQLITE_PATH = _PROJECT_ROOT / "data" / "emafrs.db"


def _make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
engine = _make_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def configure_database(database_url: str) -> None:
    """Repoints the module-level engine/session factory at a different database --
    intended for test isolation (a temp SQLite file per test module), so tests never
    read/write the real development database at data/emafrs.db. Must be called before
    `init_db()`/`get_session()` in whichever test module needs isolation."""
    global DATABASE_URL, engine, SessionLocal
    DATABASE_URL = database_url
    engine = _make_engine(database_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from sqlalchemy import inspect, text

    from emafrs.db.models import Base

    if DATABASE_URL.startswith("sqlite"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)

    # `create_all` only creates missing TABLES, never alters existing ones -- this project
    # has no formal migration tool, so new nullable columns added to an existing table
    # (e.g. signal_consistency) are added here explicitly, idempotently, rather than
    # silently missing from any database file created before that column existed.
    inspector = inspect(engine)
    if "predictions" in inspector.get_table_names():
        existing_columns = {c["name"] for c in inspector.get_columns("predictions")}
        if "signal_consistency" not in existing_columns:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE predictions ADD COLUMN signal_consistency FLOAT"))


def get_session() -> Session:
    return SessionLocal()
