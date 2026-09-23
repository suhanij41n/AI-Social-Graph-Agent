"""Reusable parametrized Cypher queries."""
from __future__ import annotations


def get_person_profile(session, person_id: str) -> dict | None:
    result = session.run(
        """
        MATCH (p:Person {id: $id})
        OPTIONAL MATCH (p)-[:MEMBER_OF]->(c:Community)
        OPTIONAL MATCH (p)-[:INTERESTED_IN]->(i:Interest)
        OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
        RETURN p AS person,
               collect(DISTINCT c.name) AS communities,
               collect(DISTINCT i.name) AS interests,
               collect(DISTINCT s.name) AS skills
        """,
        id=person_id,
    ).single()
    if result is None or result["person"] is None:
        return None
    person = dict(result["person"])
    person["communities"] = [c for c in result["communities"] if c]
    person["interests"] = [i for i in result["interests"] if i]
    person["skills"] = [s for s in result["skills"] if s]
    return person


def search_people_by_name(session, name_query: str, limit: int = 10) -> list[dict]:
    result = session.run(
        """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS toLower($q)
        RETURN p.id AS id, p.name AS name, p.city AS city, p.role AS role
        LIMIT $limit
        """,
        q=name_query, limit=limit,
    )
    return [dict(r) for r in result]


def search_people_by_filters(
    session, city: str | None = None, interest_name: str | None = None, limit: int = 50
) -> list[dict]:
    clauses = []
    params: dict = {"limit": limit}
    match = "MATCH (p:Person)"
    if interest_name:
        match += "-[:INTERESTED_IN]->(i:Interest)"
        clauses.append("toLower(i.name) = toLower($interest_name)")
        params["interest_name"] = interest_name
    if city:
        clauses.append("toLower(p.city) = toLower($city)")
        params["city"] = city
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        {match}
        {where}
        RETURN DISTINCT p.id AS id, p.name AS name, p.city AS city, p.role AS role, p.bio AS bio
        LIMIT $limit
    """
    return [dict(r) for r in session.run(query, **params)]


def find_direct_connections(session, person_id: str) -> list[dict]:
    result = session.run(
        """
        MATCH (p:Person {id: $id})-[r]-(other:Person)
        WHERE type(r) IN ['KNOWS','FRIENDS_WITH','WORKED_WITH','COLLABORATED_WITH','FOLLOWS','MET_AT']
        RETURN other.id AS id, other.name AS name, type(r) AS rel_type,
               r.strength AS strength, r.since AS since, r.last_interaction AS last_interaction
        """,
        id=person_id,
    )
    return [dict(r) for r in result]


def find_shared_interests(session, person_a: str, person_b: str) -> list[str]:
    result = session.run(
        """
        MATCH (a:Person {id: $a})-[:INTERESTED_IN]->(i:Interest)<-[:INTERESTED_IN]-(b:Person {id: $b})
        RETURN DISTINCT i.name AS name
        """,
        a=person_a, b=person_b,
    )
    return [r["name"] for r in result]


def find_shared_communities(session, person_a: str, person_b: str) -> list[str]:
    result = session.run(
        """
        MATCH (a:Person {id: $a})-[:MEMBER_OF]->(c:Community)<-[:MEMBER_OF]-(b:Person {id: $b})
        RETURN DISTINCT c.name AS name
        """,
        a=person_a, b=person_b,
    )
    return [r["name"] for r in result]


def get_person_by_id(session, person_id: str) -> dict | None:
    result = session.run("MATCH (p:Person {id: $id}) RETURN p AS person", id=person_id).single()
    return dict(result["person"]) if result else None


def get_temporal_relationships(session, person_id: str, days: int = 30) -> list[dict]:
    result = session.run(
        """
        MATCH (p:Person {id: $id})-[r]-(other:Person)
        WHERE type(r) IN ['KNOWS','FRIENDS_WITH','WORKED_WITH','COLLABORATED_WITH','FOLLOWS','MET_AT']
          AND date(r.last_interaction) >= date() - duration({days: $days})
        RETURN other.id AS id, other.name AS name, type(r) AS rel_type,
               r.last_interaction AS last_interaction
        ORDER BY r.last_interaction DESC
        """,
        id=person_id, days=days,
    )
    return [dict(r) for r in result]


def get_people_in_city(session, city: str, interest_name: str | None = None) -> list[dict]:
    return search_people_by_filters(session, city=city, interest_name=interest_name)


def get_names_for_ids(session, person_ids: list[str]) -> dict[str, str]:
    if not person_ids:
        return {}
    result = session.run(
        "MATCH (p:Person) WHERE p.id IN $ids RETURN p.id AS id, p.name AS name",
        ids=list(set(person_ids)),
    )
    return {r["id"]: r["name"] for r in result}


def get_last_interaction(session, person_a: str, person_b: str) -> str | None:
    """Most recent last_interaction date (ISO string) on any direct edge between two people."""
    result = session.run(
        """
        MATCH (a:Person {id: $a})-[r]-(b:Person {id: $b})
        WHERE type(r) IN ['KNOWS','FRIENDS_WITH','WORKED_WITH','COLLABORATED_WITH','FOLLOWS','MET_AT']
          AND r.last_interaction IS NOT NULL
        RETURN r.last_interaction AS last_interaction
        ORDER BY r.last_interaction DESC
        LIMIT 1
        """,
        a=person_a, b=person_b,
    ).single()
    return result["last_interaction"] if result else None
