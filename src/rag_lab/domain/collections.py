"""Collections: the isolation boundary. Every query is scoped to exactly one collection."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Collection:
    id: UUID
    name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(cls, name: str) -> "Collection":
        cleaned = name.strip()
        if not cleaned:
            raise ValueError("collection name is required")
        return cls(id=uuid4(), name=cleaned)
