"""~30 benchmark queries across the 9 required categories (spec §20).

Gold answers are computed directly from the live graph/DB at benchmark-build
time (via the same NetworkX/Cypher primitives the agent itself uses), not
hand-typed IDs — this keeps the benchmark correct even if the generator seed
or graph structure changes, and keeps it self-consistent with the actual
deliberately-injected bridges/hubs/recent-activity in the synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.analytics.algorithms import shortest_path
from app.analytics.bridge_detection import find_bridge_people
from app.graph.export import export_community_names, export_person_community_memberships
from app.graph.queries import get_temporal_relationships


@dataclass
class BenchmarkQuery:
    id: str
    category: str
    query_text: str
    user_id: str = "p_0001"
    gold_person_ids: list[str] = field(default_factory=list)
    gold_path: list[str] | None = None
    min_expected_results: int = 1


def _people_with_interest_in_city(session, interest_like: str, city: str) -> list[str]:
    result = session.run(
        """
        MATCH (p:Person)-[:INTERESTED_IN]->(i:Interest)
        WHERE toLower(i.name) CONTAINS toLower($interest_like) AND toLower(p.city) = toLower($city)
        RETURN DISTINCT p.id AS id
        """,
        interest_like=interest_like, city=city,
    )
    return [r["id"] for r in result]


def _people_with_interest(session, interest_like: str) -> list[str]:
    result = session.run(
        """
        MATCH (p:Person)-[:INTERESTED_IN]->(i:Interest)
        WHERE toLower(i.name) CONTAINS toLower($interest_like)
        RETURN DISTINCT p.id AS id
        """,
        interest_like=interest_like,
    )
    return [r["id"] for r in result]


def _first_degree_with_both_interests(session, user_id: str, interest_a: str, interest_b: str) -> list[str]:
    result = session.run(
        """
        MATCH (u:Person {id: $user_id})-[]-(p:Person)
        WHERE (p)-[:INTERESTED_IN]->(:Interest {name: $a})
          AND (p)-[:INTERESTED_IN]->(:Interest {name: $b})
        RETURN DISTINCT p.id AS id
        """,
        user_id=user_id, a=interest_a, b=interest_b,
    )
    return [r["id"] for r in result]


def build_benchmark_queries(session, g, user_id: str = "p_0001") -> list[BenchmarkQuery]:
    queries: list[BenchmarkQuery] = []
    names = export_community_names(session)
    memberships = export_person_community_memberships(session)

    # --- relationship (3) ---
    queries.append(BenchmarkQuery(
        id="rel_01", category="relationship", query_text="How am I connected to Maya Krishnan?",
        user_id=user_id, gold_path=shortest_path(g, user_id, "p_0002"),
    ))
    other_hubs = session.run(
        "MATCH (p:Person)-[:KNOWS|FRIENDS_WITH]-(u:Person {id:$id}) RETURN p.id AS id LIMIT 1", id=user_id
    ).single()
    if other_hubs:
        queries.append(BenchmarkQuery(
            id="rel_02", category="relationship", query_text="How am I connected to this person?",
            user_id=user_id, gold_path=shortest_path(g, user_id, other_hubs["id"]),
        ))
    queries.append(BenchmarkQuery(
        id="rel_03", category="relationship", query_text="What is my shortest path to a random distant person?",
        user_id=user_id, gold_path=shortest_path(g, user_id, "p_0300"),
    ))

    # --- community (3) ---
    user_communities = [c["community"] for c in
                        session.run("MATCH (u:Person {id:$id})-[:MEMBER_OF]->(c:Community) RETURN c.name AS community",
                                    id=user_id)]
    queries.append(BenchmarkQuery(
        id="comm_01", category="community", query_text="What communities am I connected to?",
        user_id=user_id, gold_person_ids=[], min_expected_results=1,
    ))
    queries.append(BenchmarkQuery(
        id="comm_02", category="community", query_text="What communities exist in the network?",
        user_id=user_id, min_expected_results=len(names),
    ))
    ai_members = [p for p, comms in memberships.items()
                  if next((cid for cid, n in names.items() if n == "Bangalore AI Community"), None) in comms]
    queries.append(BenchmarkQuery(
        id="comm_03", category="community", query_text="Who are the members of the Bangalore AI Community?",
        user_id=user_id, gold_person_ids=ai_members, min_expected_results=1,
    ))

    # --- bridge_detection (4) ---
    bridge_pairs = [
        ("Contemporary Dance Circle", "Independent Musicians Network"),
        ("Fashion Creators", "Photography Collective"),
        ("Bangalore AI Community", "Fashion Creators"),
        ("Startup Founders", "Bangalore AI Community"),
    ]
    id_by_name = {v: k for k, v in names.items()}
    for i, (a, b) in enumerate(bridge_pairs, start=1):
        ca, cb = id_by_name.get(a), id_by_name.get(b)
        gold = []
        if ca and cb:
            gold = [r.person_id for r in find_bridge_people(g, memberships, ca, cb, top_k=5)]
        queries.append(BenchmarkQuery(
            id=f"bridge_{i:02d}", category="bridge_detection",
            query_text=f"Who connects my {a} and {b} communities?",
            user_id=user_id, gold_person_ids=gold, min_expected_results=1,
        ))

    # --- goal_based_discovery (4) — open-ended, evaluated on non-empty + plausible top result ---
    for i, q in enumerate([
        "I'm looking for someone who could help me build a fashion + AI project.",
        "I'm looking for a photography collaborator in Bangalore.",
        "Who could help me produce an independent music album?",
        "I need a collaborator for a travel photography series.",
    ], start=1):
        queries.append(BenchmarkQuery(
            id=f"goal_{i:02d}", category="goal_based_discovery", query_text=q,
            user_id=user_id, min_expected_results=1,
        ))

    # --- outside_network_discovery (3) ---
    for i, q in enumerate([
        "Find people relevant to my goal who aren't directly connected to me.",
        "Who outside my network is interested in beauty?",
        "Who could help with a wellness app, even if I don't know them yet?",
    ], start=1):
        queries.append(BenchmarkQuery(
            id=f"outside_{i:02d}", category="outside_network_discovery", query_text=q,
            user_id=user_id, min_expected_results=1,
        ))

    # --- semantic_similarity (3), gold = ground truth INTERESTED_IN match ---
    for i, (q, interest) in enumerate([
        ("Who is interested in beauty?", "Beauty"),
        ("Who is passionate about entrepreneurship?", "Entrepreneurship"),
        ("Who is into wellness and yoga?", "Wellness"),
    ], start=1):
        gold = _people_with_interest(session, interest)
        queries.append(BenchmarkQuery(
            id=f"sem_{i:02d}", category="semantic_similarity", query_text=q,
            user_id=user_id, gold_person_ids=gold, min_expected_results=1,
        ))

    # --- location (3) ---
    for i, (q, interest, city) in enumerate([
        ("Who interested in music is near Bangalore?", "Music", "Bangalore"),
        ("Who interested in dance is in Mumbai?", "Dance", "Mumbai"),
        ("Who interested in travel is in Goa?", "Travel", "Goa"),
    ], start=1):
        gold = _people_with_interest_in_city(session, interest, city)
        queries.append(BenchmarkQuery(
            id=f"loc_{i:02d}", category="location", query_text=q,
            user_id=user_id, gold_person_ids=gold, min_expected_results=1,
        ))

    # --- temporal (3) ---
    recent = [r["id"] for r in get_temporal_relationships(session, user_id, days=365)]
    queries.append(BenchmarkQuery(
        id="temp_01", category="temporal", query_text="Who have I recently become connected to?",
        user_id=user_id, gold_person_ids=recent, min_expected_results=0,
    ))
    queries.append(BenchmarkQuery(
        id="temp_02", category="temporal", query_text="Who did I collaborate with recently?",
        user_id=user_id, gold_person_ids=recent, min_expected_results=0,
    ))
    queries.append(BenchmarkQuery(
        id="temp_03", category="temporal", query_text="Which communities have become more important recently?",
        user_id=user_id, min_expected_results=1,
    ))

    # --- cross_domain (4) ---
    cross_pairs = [
        ("photography", "Photography", "Travel"),
        ("AI + fashion", "AI", "Beauty"),
        ("dance + music", "Contemporary", "Pop"),
        ("food + travel", "Food", "Travel"),
    ]
    for i, (label, a, b) in enumerate(cross_pairs, start=1):
        gold = _first_degree_with_both_interests(session, user_id, a, b)
        queries.append(BenchmarkQuery(
            id=f"cross_{i:02d}", category="cross_domain",
            query_text=f"Who in my network is interested in both {a} and {b}?",
            user_id=user_id, gold_person_ids=gold, min_expected_results=0,
        ))

    return queries
