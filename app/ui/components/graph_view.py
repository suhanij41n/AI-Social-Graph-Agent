"""Interactive graph visualization highlighting the user, recommended people,
paths, communities, and bridge nodes.

Uses streamlit-agraph (a proper Streamlit component with a bundled frontend)
rather than pyvis, since pyvis's generate_html() references vis.js via a
relative node_modules path that only resolves in a Jupyter context — it
breaks when embedded via components.html with no internet access to fall
back on. streamlit-agraph ships its compiled frontend inside the package,
so it renders reliably offline.
"""
from __future__ import annotations

import networkx as nx
from streamlit_agraph import Config, Edge, Node, agraph

USER_COLOR = "#e63946"
CANDIDATE_COLOR = "#2a9d8f"
PATH_COLOR = "#f4a261"
BRIDGE_COLOR = "#8338ec"
DEFAULT_COLOR = "#adb5bd"


def render_graph(
    g: nx.Graph,
    user_id: str,
    candidate_ids: list[str],
    path_ids: list[str] | None = None,
    bridge_ids: list[str] | None = None,
    name_lookup: dict[str, str] | None = None,
    max_neighbors: int = 60,
    height: int = 550,
) -> None:
    path_ids = path_ids or []
    bridge_ids = bridge_ids or []
    name_lookup = name_lookup or {}

    focus_nodes = set([user_id]) | set(candidate_ids) | set(path_ids) | set(bridge_ids)
    for node in list(focus_nodes):
        if node in g:
            focus_nodes |= set(list(g.neighbors(node))[:8])

    keep = {n for n in focus_nodes if n in g}
    if len(keep) > max_neighbors:
        keep = set(list(keep)[:max_neighbors])
    sub = g.subgraph(keep)

    path_id_set = set(path_ids)

    nodes = []
    for node_id in sub.nodes():
        label = name_lookup.get(node_id, node_id)
        if node_id == user_id:
            color, size = USER_COLOR, 28
        elif node_id in bridge_ids:
            color, size = BRIDGE_COLOR, 24
        elif node_id in path_id_set:
            color, size = PATH_COLOR, 20
        elif node_id in candidate_ids:
            color, size = CANDIDATE_COLOR, 20
        else:
            color, size = DEFAULT_COLOR, 12
        nodes.append(Node(id=node_id, label=label, color=color, size=size, title=label))

    path_edges = set(zip(path_ids, path_ids[1:]))
    edges = []
    for a, b, data in sub.edges(data=True):
        is_path_edge = (a, b) in path_edges or (b, a) in path_edges
        edges.append(Edge(
            source=a, target=b,
            color=PATH_COLOR if is_path_edge else "#495057",
            width=3 if is_path_edge else 1,
        ))

    config = Config(height=height, width="100%", directed=False, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)
