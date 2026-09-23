from app.analytics.algorithms import (
    betweenness_centrality,
    degree_centrality,
    degree_of_connection,
    detect_communities,
    pagerank,
    shortest_path,
)
from app.analytics.bridge_detection import find_bridge_people, top_bridge_people_overall


def test_shortest_path_matches_hand_computed(fixture_networkx_graph):
    path = shortest_path(fixture_networkx_graph, "f01", "f04")
    assert path is not None
    assert path[0] == "f01"
    assert path[-1] == "f04"
    assert len(path) - 1 == 2  # hand-computed: f01-f02-f04 or f01-f03-f04


def test_shortest_path_no_path_for_isolated_person(fixture_networkx_graph):
    assert shortest_path(fixture_networkx_graph, "f01", "f20") is None


def test_degree_of_connection(fixture_networkx_graph):
    assert degree_of_connection(fixture_networkx_graph, "f01", "f01") == 0
    assert degree_of_connection(fixture_networkx_graph, "f01", "f20") is None


def test_bridge_person_has_highest_betweenness(fixture_networkx_graph):
    scores = betweenness_centrality(fixture_networkx_graph)
    top_person = max(scores, key=scores.get)
    assert top_person == "f17"


def test_degree_centrality_runs_and_is_bounded(fixture_networkx_graph):
    scores = degree_centrality(fixture_networkx_graph)
    assert set(scores) == set(fixture_networkx_graph.nodes())
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_pagerank_sums_to_one(fixture_networkx_graph):
    scores = pagerank(fixture_networkx_graph)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_detect_communities_finds_at_least_two_clusters(fixture_networkx_graph):
    partition = detect_communities(fixture_networkx_graph)
    assert len(set(partition.values())) >= 2
    # the two dense cliques should mostly land in different clusters
    a_cluster = partition["f01"]
    b_cluster = partition["f09"]
    assert a_cluster != b_cluster


def test_find_bridge_people_identifies_fixture_bridge(fixture_networkx_graph, fixture_memberships):
    results = find_bridge_people(fixture_networkx_graph, fixture_memberships, "fc_a", "fc_b")
    assert len(results) >= 1
    assert results[0].person_id == "f17"
    assert set(results[0].communities_bridged) == {"fc_a", "fc_b"}


def test_top_bridge_people_overall_includes_fixture_bridge(fixture_networkx_graph, fixture_memberships):
    results = top_bridge_people_overall(fixture_networkx_graph, fixture_memberships)
    ids = [r["person_id"] for r in results]
    assert "f17" in ids
