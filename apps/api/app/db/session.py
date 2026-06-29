from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Base

_ENGINES: dict[str, Engine] = {}


def _sqlite_connect_args(url: str) -> dict[str, bool]:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def get_engine() -> Engine:
    database_url = get_settings().database_url
    engine = _ENGINES.get(database_url)
    if engine is None:
        engine = create_engine(
            database_url,
            future=True,
            connect_args=_sqlite_connect_args(database_url),
        )
        _ENGINES[database_url] = engine
    return engine


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def get_session() -> Generator[Session, None, None]:
    init_db()
    session_factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
