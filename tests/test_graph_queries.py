import pytest

pytestmark = pytest.mark.integration


def test_get_person_profile(neo4j_fixture_session):
    from app.graph.queries import get_person_profile

    profile = get_person_profile(neo4j_fixture_session, "f17")
    assert profile is not None
    assert profile["name"] == "Fixture Bridge"
    assert set(profile["communities"]) == {"Fixture Community A", "Fixture Community B"}
    assert "Fixture Dance" in profile["interests"]
    assert "Fixture Music" in profile["interests"]


def test_get_person_profile_missing_returns_none(neo4j_fixture_session):
    from app.graph.queries import get_person_profile

    assert get_person_profile(neo4j_fixture_session, "does_not_exist") is None


def test_search_people_by_name(neo4j_fixture_session):
    from app.graph.queries import search_people_by_name

    results = search_people_by_name(neo4j_fixture_session, "Fixture A1")
    assert any(r["id"] == "f01" for r in results)


def test_find_direct_connections(neo4j_fixture_session):
    from app.graph.queries import find_direct_connections

    connections = find_direct_connections(neo4j_fixture_session, "f01")
    ids = {c["id"] for c in connections}
    assert "f02" in ids
    assert "f03" in ids


def test_find_shared_interests(neo4j_fixture_session):
    from app.graph.queries import find_shared_interests

    shared = find_shared_interests(neo4j_fixture_session, "f01", "f17")
    assert "Fixture Dance" in shared


def test_find_shared_communities(neo4j_fixture_session):
    from app.graph.queries import find_shared_communities

    shared = find_shared_communities(neo4j_fixture_session, "f01", "f17")
    assert "Fixture Community A" in shared


def test_find_shared_communities_none_for_unrelated(neo4j_fixture_session):
    from app.graph.queries import find_shared_communities

    shared = find_shared_communities(neo4j_fixture_session, "f01", "f20")
    assert shared == []
