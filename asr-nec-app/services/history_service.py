from __future__ import annotations

from sqlmodel import Session, select

from models.entities import Correction


def list_corrections(session: Session, limit: int = 100) -> list[Correction]:
    statement = (
        select(Correction).order_by(Correction.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
    )
    return list(session.exec(statement).all())


def get_correction(session: Session, correction_id: str) -> Correction | None:
    return session.get(Correction, correction_id)
