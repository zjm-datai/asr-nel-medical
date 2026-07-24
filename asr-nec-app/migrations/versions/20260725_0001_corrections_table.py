"""create corrections table

Revision ID: 0001
Revises:
Create Date: 2026-07-25

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "corrections",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("audio_path", sa.String(length=1000), nullable=False),
        sa.Column("audio_url", sa.String(length=1000), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("asr_text", sa.Text(), nullable=False),
        sa.Column("corrected_text", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("candidates", sa.JSON(), nullable=False),
        sa.Column("timings", sa.JSON(), nullable=False),
    )
    op.create_index("ix_corrections_source", "corrections", ["source"])


def downgrade() -> None:
    op.drop_index("ix_corrections_source", table_name="corrections")
    op.drop_table("corrections")
