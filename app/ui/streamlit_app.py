"""Streamlit UI for the AI Social Graph Agent (spec §19).

Run: streamlit run app/ui/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from app.agent.graph_builder import run_agent  # noqa: E402
from app.graph.connection import get_session  # noqa: E402
from app.graph.export import export_person_graph  # noqa: E402
from app.graph.queries import get_person_profile  # noqa: E402
from app.ui.components.evidence_panel import render_evidence  # noqa: E402
from app.ui.components.graph_view import render_graph  # noqa: E402
from app.ui.components.person_detail import render_person_detail  # noqa: E402
from app.ui.components.why_not_panel import render_why_not_panel  # noqa: E402

st.set_page_config(page_title="AI Social Graph Agent", layout="wide")

DEMO_USER_ID = "p_0001"

EXAMPLE_QUERIES = [
    "I'm looking for someone who could help me build a fashion + AI project.",
    "What communities am I connected to?",
    "Who connects my dance and music communities?",
    "How am I connected to Maya Krishnan?",
    "Find people relevant to my goal who aren't directly connected to me.",
    "Who in my network is interested in both photography and travel?",
    "Who interested in music is near Bangalore?",
    "Who have I recently become connected to?",
]


@st.cache_resource
def get_graph_and_names():
    with get_session() as session:
        g = export_person_graph(session)
        rows = session.run("MATCH (p:Person) RETURN p.id AS id, p.name AS name")
        names = {r["id"]: r["name"] for r in rows}
    return g, names


@st.cache_resource
def get_all_people():
    with get_session() as session:
        rows = session.run(
            "MATCH (p:Person) RETURN p.id AS id, p.name AS name, p.city AS city, "
            "p.role AS role, p.country AS country ORDER BY p.name"
        )
        return [dict(r) for r in rows]


@st.cache_resource
def get_demo_user_profile():
    with get_session() as session:
        return get_person_profile(session, DEMO_USER_ID)


def render_people_browser(names: dict[str, str], preselect: str | None = None) -> None:
    """Search/browse all 300 people, then drill into one person's full profile."""
    people = get_all_people()

    search = st.text_input("Search people by name", key="people_search",
                            placeholder="e.g. Maya, or leave blank to see everyone")
    filtered = [p for p in people if search.lower() in p["name"].lower()] if search else people

    st.caption(f"{len(filtered)} of {len(people)} people")
    st.dataframe(
        [{"Name": p["name"], "City": p["city"], "Country": p["country"], "Role": p["role"]} for p in filtered],
        use_container_width=True, height=250,
    )

    options = [p["id"] for p in filtered] or [DEMO_USER_ID]
    default_index = options.index(preselect) if preselect in options else 0
    selected = st.selectbox(
        "View full profile", options, index=default_index,
        format_func=lambda p: f"{names.get(p, p)}" + (" (you)" if p == DEMO_USER_ID else ""),
        key="people_browser_select",
    )
    with get_session() as session:
        render_person_detail(session, selected)


def main() -> None:
    st.title("🕸️ AI Social Graph Agent")
    st.caption(
        "Ask about relationships, communities, bridges, or goal-based discovery across a "
        "300-person synthetic social network spanning AI, dance, fashion, photography, "
        "music, travel, and more."
    )

    g, names = get_graph_and_names()
    demo_profile = get_demo_user_profile()

    if demo_profile:
        st.info(
            f"👤 **You are {demo_profile['name']}** — {demo_profile['role']} in "
            f"{demo_profile['city']}, {demo_profile['country']}. {demo_profile['bio']}"
        )

    with st.sidebar:
        st.subheader("Try an example")
        for q in EXAMPLE_QUERIES:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state["query_input"] = q

    query = st.text_input("Ask the agent a question", key="query_input",
                           placeholder="e.g. Who could help me with an AI + fashion project?")
    run_clicked = st.button("Ask", type="primary")

    if run_clicked and query.strip():
        with st.spinner("Reasoning over the graph..."):
            result = run_agent(query, user_id=DEMO_USER_ID)
        st.session_state["last_result"] = result
        st.session_state["last_query"] = query

    result = st.session_state.get("last_result")

    tab_ranked, tab_graph, tab_people, tab_why_not = st.tabs(
        ["Ranked candidates", "Graph view", "People", "Why not?"]
    )

    if result is None:
        with tab_ranked:
            st.info("Ask a question or pick an example from the sidebar to get started.")
        with tab_graph:
            st.info("Ask a question first to see a graph centered on the results.")
        with tab_people:
            render_people_browser(names, preselect=DEMO_USER_ID)
        with tab_why_not:
            st.info("Ask a question first, then come back here to compare two of the results.")
        return

    st.markdown("## Answer")
    st.write(result.get("final_response", ""))
    st.caption(f"Intent: `{result.get('intent')}` · Tools used: {', '.join(result.get('tool_calls_log', []))}")

    ranked = result.get("ranked_candidates", [])
    pool = result.get("graph_evidence", {}).get("_candidate_pool", {})
    bundles = result.get("graph_evidence", {}).get("_evidence_bundles", {})

    with tab_ranked:
        if not ranked:
            st.write("No ranked candidates for this query.")
        else:
            st.markdown("### Ranked candidates")
            table_rows = []
            for rc in ranked:
                pid = rc["person_id"]
                cand = pool.get(pid)
                key_reason = ""
                bundle = bundles.get(pid)
                if bundle and bundle.summary_lines():
                    key_reason = bundle.summary_lines()[0]
                table_rows.append({
                    "Name": cand.name if cand else pid,
                    "Score": round(rc["score"], 3),
                    "Network distance": cand.network_degree if cand else None,
                    "Key reason": key_reason,
                })
            st.dataframe(table_rows, use_container_width=True)

            st.markdown("### Evidence")
            for rc in ranked:
                pid = rc["person_id"]
                cand = pool.get(pid)
                name = cand.name if cand else pid
                render_evidence(name, rc["score"], bundles.get(pid), rc.get("normalized_features", {}))

    with tab_graph:
        candidate_ids = [rc["person_id"] for rc in ranked]
        path_ids = []
        for pid in candidate_ids:
            cand = pool.get(pid)
            if cand and cand.path:
                path_ids = cand.path
                break
        bridge_ids = [b["person_id"] for b in result.get("graph_evidence", {}).get("bridge_people", [])]
        render_graph(g, DEMO_USER_ID, candidate_ids, path_ids=path_ids, bridge_ids=bridge_ids, name_lookup=names)

    with tab_people:
        preselect = ranked[0]["person_id"] if ranked else DEMO_USER_ID
        render_people_browser(names, preselect=preselect)

    with tab_why_not:
        render_why_not_panel(DEMO_USER_ID, ranked, names)


if __name__ == "__main__":
    main()
