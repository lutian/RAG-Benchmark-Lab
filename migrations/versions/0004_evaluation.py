"""Evaluation: golden questions with evidence spans, and benchmark runs.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-01
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "golden_questions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_golden_questions_collection_id", "golden_questions", ["collection_id"])
    op.create_table(
        "golden_evidence",
        sa.Column(
            "question_id",
            sa.Uuid(),
            sa.ForeignKey("golden_questions.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), primary_key=True),
        # No cascade: deleting a labelled document must fail, not empty its question.
        # Deferred so a collection delete (cascading in any order) passes at commit.
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", deferrable=True, initially="DEFERRED"),
            nullable=False,
        ),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset", name="ck_evidence_span"
        ),
    )
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "collection_id",
            sa.Uuid(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_benchmark_runs_collection_id", "benchmark_runs", ["collection_id"])


def downgrade() -> None:
    op.drop_table("benchmark_runs")
    op.drop_table("golden_evidence")
    op.drop_table("golden_questions")
