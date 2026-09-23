"""Renders evidence for a ranked candidate: path, shared interests/communities,
semantic relevance, and ranking metrics."""
from __future__ import annotations

import streamlit as st


def render_evidence(person_name: str, score: float, bundle, normalized_features: dict) -> None:
    with st.expander(f"{person_name} — score {score:.2f}"):
        if bundle is None or bundle.is_empty():
            st.write("No supporting evidence.")
            return

        if bundle.facts:
            st.markdown("**Database facts**")
            for f in bundle.facts:
                st.write(f"- {f.description}")

        if bundle.metrics:
            st.markdown("**Calculated metrics**")
            for m in bundle.metrics:
                st.write(f"- {m.description}")

        if bundle.semantic:
            st.markdown("**Semantic relevance**")
            for s in bundle.semantic:
                st.write(f"- {s.description}")

        st.markdown("**Ranking signal breakdown (normalized 0-1)**")
        st.bar_chart(normalized_features)
