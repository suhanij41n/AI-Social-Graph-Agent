"""Community bridge detection (spec §14 — signature feature).

Combines betweenness centrality with community membership to find people who
connect otherwise-separate communities, and explains which communities they
connect and via which relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from app.analytics.algorithms import betweenness_centrality


@dataclass
class BridgeResult:
    person_id: str
    betweenness: float
    bridge_strength: float  # pair-specific: how strongly tied into BOTH sides
    communities_bridged: list[str]
    connecting_edges: list[dict] = field(default_factory=list)


def find_bridge_people(
    g: nx.Graph,
    memberships: dict[str, set[str]],
    community_a: str,
    community_b: str,
    top_k: int = 5,
) -> list[BridgeResult]:
    """Find people who structurally and/or via membership bridge two communities.

    A bridge candidate is anyone who either:
      (a) is a member of both community_a and community_b, or
      (b) has direct edges into both communities (structural bridge without
          formal dual membership).

    Ranked by a pair-specific bridge_strength = min(weighted ties into A,
    weighted ties into B) — a bottleneck metric requiring real presence on
    BOTH sides, rather than raw global betweenness. Global betweenness alone
    would over-favor generic high-degree hubs whose many connections are
    mostly unrelated to either community, over someone deliberately embedded
    in both (e.g. a dance-and-music bridge with modest overall degree).
    Global betweenness is still reported for reference/explanation.
    """
    betweenness = betweenness_centrality(g)

    members_a = {p for p, comms in memberships.items() if community_a in comms}
    members_b = {p for p, comms in memberships.items() if community_b in comms}

    candidates: dict[str, BridgeResult] = {}

    def _ties_into(person: str, group: set[str]) -> float:
        return sum(
            g.get_edge_data(person, n).get("weight", 0.3)
            for n in g.neighbors(person) if n in group
        )

    # (a) dual membership
    for person in members_a & members_b:
        ties_a = _ties_into(person, members_a - {person})
        ties_b = _ties_into(person, members_b - {person})
        candidates[person] = BridgeResult(
            person_id=person,
            betweenness=betweenness.get(person, 0.0),
            bridge_strength=min(ties_a, ties_b) + 0.5,  # dual-membership bonus
            communities_bridged=[community_a, community_b],
        )

    # (b) structural bridges: not a member of either, but directly connected to both
    for node in g.nodes():
        if node in candidates or node in members_a or node in members_b:
            continue
        ties_a = _ties_into(node, members_a)
        ties_b = _ties_into(node, members_b)
        if ties_a > 0 and ties_b > 0:
            candidates[node] = BridgeResult(
                person_id=node,
                betweenness=betweenness.get(node, 0.0),
                bridge_strength=min(ties_a, ties_b),
                communities_bridged=[community_a, community_b],
            )

    for person, result in candidates.items():
        edges = []
        for neighbor in g.neighbors(person):
            if neighbor in members_a or neighbor in members_b:
                edge_data = g.get_edge_data(person, neighbor)
                edges.append({
                    "to": neighbor,
                    "weight": edge_data.get("weight"),
                    "types": sorted(edge_data.get("types", [])),
                })
        result.connecting_edges = edges

    ranked = sorted(candidates.values(), key=lambda r: (r.bridge_strength, r.betweenness), reverse=True)
    return ranked[:top_k]


def top_bridge_people_overall(
    g: nx.Graph, memberships: dict[str, set[str]], top_k: int = 10
) -> list[dict]:
    """People whose betweenness is high AND who span 2+ communities (general bridges,
    not tied to a specific community pair)."""
    betweenness = betweenness_centrality(g)
    results = []
    for person, score in betweenness.items():
        comms = memberships.get(person, set())
        if len(comms) >= 2:
            results.append({"person_id": person, "betweenness": score, "communities": sorted(comms)})
    results.sort(key=lambda r: r["betweenness"], reverse=True)
    return results[:top_k]
