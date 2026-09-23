"""Neo4j driver singleton and session context manager."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from typing import Iterator

from neo4j import Driver, GraphDatabase

from app.config import get_settings

_driver: Driver | None = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        atexit.register(close_driver)
    return _driver


@contextmanager
def get_session() -> Iterator:
    driver = get_driver()
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
