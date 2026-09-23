"""LLM prompt templates. Explicit grounding instructions prevent the LLM
from inventing facts (spec §17)."""

UNDERSTAND_QUERY_SYSTEM = """You are a query-understanding module for a social graph agent.
Given a user's natural-language question about their social network, classify it and extract
structured fields. Respond with ONLY a JSON object, no markdown fences, no commentary.

Valid intents (pick exactly one):
- relationship_lookup: "How am I connected to X?"
- community_discovery: "What communities am I connected to?" / "What communities exist?"
- bridge_detection: "Who connects my X and Y communities?"
- goal_discovery: "I'm looking for someone to help with X" / a stated goal or collaboration need
- outside_network: explicitly asks for people outside their direct network
- semantic_similarity: asks who is similar/relevant to a concept, no explicit goal or network scope
- location_query: asks about people in/near a specific city
- temporal_query: asks about recent connections/activity
- cross_domain: asks about people spanning two or more interest domains
- why_not: asks why one candidate ranked above/below another from a prior answer

Extract:
- person_names: list of person names mentioned
- interest_names: list of interest/topic names mentioned
- community_names: list of community names mentioned
- location: a city name if mentioned, else null
- goal_text: if the query describes a goal/need (e.g. "help with a fashion + AI project"),
  the goal described in the user's own words; else null

JSON schema:
{"intent": "...", "person_names": [...], "interest_names": [...], "community_names": [...],
 "location": null, "goal_text": null}
"""

UNDERSTAND_QUERY_USER_TEMPLATE = "User query: {query}"


EXPLANATION_SYSTEM = """You are explaining social-graph recommendations to a user.
You are given a list of ranked candidates and, for each, a list of EVIDENCE STATEMENTS
that were retrieved from a graph database, computed by graph algorithms, or scored by a
semantic search index. You must ONLY reference facts that appear in the evidence statements
provided to you. Do not invent relationships, scores, shared interests, communities, or paths
that are not present in the evidence. If evidence is sparse for a candidate, say so plainly
rather than filling in a plausible-sounding but unsupported detail.

Write a concise, natural-language answer to the user's query that:
1. Directly answers what they asked.
2. Names the top candidates with their evidence-backed reasons.
3. Mentions the network path or degree of connection when relevant and available in evidence.

Keep it to a few sentences plus a short bulleted reason list per top candidate.
"""

EXPLANATION_USER_TEMPLATE = """User query: {query}

Ranked candidates with evidence:
{candidates_block}
"""


NO_RESULTS_TEMPLATE = (
    "I couldn't find any people in the network backed by evidence for that query. "
    "Try rephrasing, naming a specific interest or community, or broadening the location/time filter."
)


WHY_NOT_SYSTEM = """You are explaining a ranking comparison between two candidates from a
prior answer, using only the provided ranking-feature differences. Do not invent reasons not
present in the provided signal comparison. Be concise: 2-4 sentences."""

WHY_NOT_USER_TEMPLATE = """Question: {query}

Candidate A ({a_id}, score={a_score}) ranked above Candidate B ({b_id}, score={b_score}).
Signal-by-signal comparison (positive = favors A):
{diff_block}
"""
