"""Person profile card: bio, interests, communities, skills."""
from __future__ import annotations

import streamlit as st

from app.graph.queries import get_person_profile


def render_person_detail(session, person_id: str) -> None:
    profile = get_person_profile(session, person_id)
    if profile is None:
        st.warning("Person not found.")
        return

    st.subheader(profile["name"])
    st.caption(f"{profile['role']} · {profile['city']}, {profile['country']} · {profile['age_range']}")
    st.write(profile["bio"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Interests**")
        for i in profile["interests"]:
            st.write(f"- {i}")
    with col2:
        st.markdown("**Communities**")
        for c in profile["communities"]:
            st.write(f"- {c}")
    with col3:
        st.markdown("**Skills**")
        for s in profile["skills"]:
            st.write(f"- {s}")
