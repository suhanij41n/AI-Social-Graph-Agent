"""'Why is X ranked above Y?' comparison panel (spec §18)."""
from __future__ import annotations

import streamlit as st

from app.agent.graph_builder import run_agent


def render_why_not_panel(user_id: str, ranked_candidates: list[dict], name_lookup: dict[str, str]) -> None:
    if len(ranked_candidates) < 2:
        st.info("Need at least 2 ranked candidates to compare.")
        return

    options = [rc["person_id"] for rc in ranked_candidates]
    labels = {pid: name_lookup.get(pid, pid) for pid in options}

    col1, col2 = st.columns(2)
    with col1:
        a_id = st.selectbox("Candidate A (higher-ranked)", options, format_func=lambda p: labels[p], key="why_not_a")
    with col2:
        remaining = [p for p in options if p != a_id]
        b_id = st.selectbox("Candidate B", remaining, format_func=lambda p: labels[p], key="why_not_b")

    if st.button("Explain ranking"):
        with st.spinner("Comparing ranking signals..."):
            result = run_agent(
                f"Why is {labels[a_id]} ranked above {labels[b_id]}?",
                user_id=user_id,
                previous_ranked_candidates=ranked_candidates,
                compare_pair=(a_id, b_id),
            )
        st.write(result["final_response"])
