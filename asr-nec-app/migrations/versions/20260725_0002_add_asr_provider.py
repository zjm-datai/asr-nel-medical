"""add ASR provider to corrections

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-25
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "corrections",
        sa.Column(
            "asr_provider",
            sa.String(length=32),
            nullable=False,
            server_default="local_whisper",
        ),
    )


def downgrade() -> None:
    op.drop_column("corrections", "asr_provider")
