from __future__ import annotations

from collections.abc import Generator
from typing import Any

from fastapi import FastAPI
from sqlalchemy import event
from sqlmodel import Session, create_engine

from configs.base import Settings, get_settings

settings = get_settings()
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=1800,
)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(
        dbapi_connection: Any, connection_record: Any
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_app(app: FastAPI, settings: Settings) -> None:
    app.state.db_engine = engine


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
