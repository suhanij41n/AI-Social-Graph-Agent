import json
import sys
from pathlib import Path

import networkx as nx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fixture_graph.json"


@pytest.fixture(scope="session")
def fixture_data() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def fixture_networkx_graph(fixture_data: dict) -> nx.Graph:
    """Build the fixture person-graph directly in NetworkX, no Neo4j required."""
    g = nx.Graph()
    for person in fixture_data["people"]:
        g.add_node(person["id"], name=person["name"])
    for edge in fixture_data["relationships"]["person_person"]:
        a, b, w = edge["from"], edge["to"], edge["strength"]
        if g.has_edge(a, b):
            g[a][b]["weight"] = max(g[a][b]["weight"], w)
            g[a][b]["types"].add(edge["type"])
        else:
            g.add_edge(a, b, weight=w, types={edge["type"]})
    return g


@pytest.fixture(scope="session")
def fixture_memberships(fixture_data: dict) -> dict[str, set[str]]:
    memberships: dict[str, set[str]] = {}
    for row in fixture_data["relationships"]["member_of"]:
        memberships.setdefault(row["person"], set()).add(row["community"])
    return memberships


@pytest.fixture
def neo4j_test_session():
    """A live Neo4j session for integration tests. Restores the full 300-person
    generated dataset at teardown regardless of what the test loaded."""
    from app.graph.connection import get_session
    from app.graph.loader import load_all

    with get_session() as session:
        yield session

    with get_session() as session:
        load_all(session, clear=True)


def _load_fixture_graph(session, fixture_data: dict) -> None:
    from app.graph.loader import (
        clear_graph, load_communities, load_has_skill, load_interested_in,
        load_member_of, load_people, load_person_person,
    )
    from app.graph.schema import apply_schema

    clear_graph(session)
    apply_schema(session)
    load_people(session, fixture_data["people"])
    load_communities(session, fixture_data["communities"])
    session.run(
        "UNWIND $rows AS row MERGE (i:Interest {id: row.id}) SET i.name = row.name, i.level = row.level",
        rows=fixture_data["interests"],
    )
    load_person_person(session, fixture_data["relationships"]["person_person"])
    load_member_of(session, fixture_data["relationships"]["member_of"])
    load_interested_in(session, fixture_data["relationships"]["interested_in"])


@pytest.fixture
def neo4j_fixture_session(fixture_data: dict):
    """A live Neo4j session pre-loaded with the small fixture graph. Restores
    the full 300-person generated dataset at teardown."""
    from app.graph.connection import get_session
    from app.graph.loader import load_all

    with get_session() as session:
        _load_fixture_graph(session, fixture_data)
        yield session

    with get_session() as session:
        load_all(session, clear=True)
