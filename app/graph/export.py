"""Export a Neo4j subgraph into a NetworkX graph for analytics.

Neo4j remains the source of truth; NetworkX is a disposable, derived view
rebuilt per session/query rather than kept in sync separately.
"""
from __future__ import annotations

import networkx as nx

PERSON_PERSON_TYPES = [
    "KNOWS", "FRIENDS_WITH", "WORKED_WITH", "COLLABORATED_WITH", "FOLLOWS", "MET_AT",
]


def export_person_graph(session) -> nx.Graph:
    """Undirected, weighted person-to-person graph (weight = relationship strength)."""
    g = nx.Graph()

    people = session.run("MATCH (p:Person) RETURN p.id AS id, p.name AS name")
    for row in people:
        g.add_node(row["id"], name=row["name"])

    rel_filter = "|".join(PERSON_PERSON_TYPES)
    edges = session.run(
        f"""
        MATCH (a:Person)-[r:{rel_filter}]->(b:Person)
        RETURN a.id AS a, b.id AS b, type(r) AS type, r.strength AS strength
        """
    )
    for row in edges:
        a, b = row["a"], row["b"]
        strength = row["strength"] if row["strength"] is not None else 0.3
        if g.has_edge(a, b):
            # keep the strongest tie type as the effective weight
            g[a][b]["weight"] = max(g[a][b]["weight"], strength)
            g[a][b]["types"].add(row["type"])
        else:
            g.add_edge(a, b, weight=strength, types={row["type"]})

    return g


def export_person_community_memberships(session) -> dict[str, set[str]]:
    """Map person_id -> set of community_ids."""
    rows = session.run(
        "MATCH (p:Person)-[:MEMBER_OF]->(c:Community) RETURN p.id AS person, c.id AS community"
    )
    memberships: dict[str, set[str]] = {}
    for row in rows:
        memberships.setdefault(row["person"], set()).add(row["community"])
    return memberships


def export_community_names(session) -> dict[str, str]:
    rows = session.run("MATCH (c:Community) RETURN c.id AS id, c.name AS name")
    return {row["id"]: row["name"] for row in rows}
