"""Run every contract test against each persistence adapter."""

import os
from collections.abc import Callable, Iterator

import pytest

from rag_lab.adapters.outbound.persistence.memory import InMemoryUnitOfWork
from rag_lab.application.ports.repositories import UnitOfWork


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.integration)])
def uow_factory(request: pytest.FixtureRequest) -> Iterator[Callable[[], UnitOfWork]]:
    if request.param == "memory":
        shared = InMemoryUnitOfWork()
        yield lambda: shared
        return

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from tests.integration.database import postgres_uow_factory

    yield from postgres_uow_factory(url)
