"""LangGraph node functions.

understand_query -> (routing via app/agent/routing.py, dispatched inside
run_tools) -> run_tools -> fuse_evidence -> rank_candidates ->
validate_evidence -> generate_explanation | no_results_response
"why not" queries route straight to why_not_response.
"""
from __future__ import annotations

import json

from app.agent import prompts, tools as agent_tools
from app.agent.evidence import CalculatedMetric, DatabaseFact, EvidenceBundle, SemanticScore
from app.agent.graph_state import AgentState
from app.agent.llm import get_llm
from app.agent.routing import ANALYTICS, GRAPH_RETRIEVAL, PATH_ANALYSIS, SEMANTIC_RETRIEVAL, TEMPORAL, tool_nodes_for_intent
from app.graph.queries import get_names_for_ids, search_people_by_name
from app.logging_config import get_logger
from app.ranking.explain import compare_candidates
from app.ranking.features import QueryContext, compute_features
from app.ranking.scorer import score_candidates
from app.retrieval.hybrid import build_candidate_pool

logger = get_logger(__name__)

TOP_K_FOR_EXPLANATION = 5


def _resolve_person_names(session, names: list[str]) -> list[str]:
    ids = []
    for name in names:
        matches = search_people_by_name(session, name, limit=1)
        if matches:
            ids.append(matches[0]["id"])
    return ids


def understand_query_node(state: AgentState) -> dict:
    session = state["_session"]
    query = state["user_query"]
    llm = get_llm()

    messages = [
        {"role": "system", "content": prompts.UNDERSTAND_QUERY_SYSTEM},
        {"role": "user", "content": prompts.UNDERSTAND_QUERY_USER_TEMPLATE.format(query=query)},
    ]
    response = llm.invoke(messages)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse intent JSON, defaulting to cross_domain: %r", response.content)
        parsed = {"intent": "cross_domain", "person_names": [], "interest_names": [],
                   "community_names": [], "location": None, "goal_text": None}

    person_ids = _resolve_person_names(session, parsed.get("person_names", []))

    entities = {
        "person_names": parsed.get("person_names", []),
        "interest_names": parsed.get("interest_names", []),
        "community_names": parsed.get("community_names", []),
        "location": parsed.get("location"),
        "goal_text": parsed.get("goal_text"),
    }
    constraints = {
        "location": parsed.get("location"),
        "max_degree": None,
        "days": 30,
        "top_k": TOP_K_FOR_EXPLANATION,
    }

    return {
        "intent": parsed.get("intent", "cross_domain"),
        "entities": entities,
        "constraints": constraints,
        "candidate_people": person_ids,
        "tool_calls_log": ["understand_query"],
        "errors": [],
    }


def run_tools_node(state: AgentState) -> dict:
    """Dispatches only the tool groups relevant to the classified intent."""
    session = state["_session"]
    g = state["_graph"]
    user_id = state["user_id"]
    intent = state["intent"]
    entities = state["entities"]
    constraints = state["constraints"]

    active_nodes = tool_nodes_for_intent(intent)
    log = list(state.get("tool_calls_log", []))

    graph_evidence: dict = {}
    semantic_evidence: dict = {}
    candidate_people = list(state.get("candidate_people", []))

    query_text = entities.get("goal_text") or state["user_query"]

    if GRAPH_RETRIEVAL in active_nodes:
        log.append("search_people")
        results = agent_tools.search_people(
            session, city=entities.get("location"),
            interest_name=(entities.get("interest_names") or [None])[0],
        )
        graph_evidence["filtered_people"] = results
        candidate_people += [r["id"] for r in results if r["id"] != user_id]

    if SEMANTIC_RETRIEVAL in active_nodes:
        log.append("semantic_search")
        hits = agent_tools.semantic_search(query_text, entity_type="people", top_k=20)
        semantic_evidence["semantic_hits"] = hits
        candidate_people += [h["entity_id"] for h in hits if h["entity_id"] != user_id]

    if ANALYTICS in active_nodes:
        log.append("analyze_community / find_bridge_people")
        community_names = entities.get("community_names", [])
        if len(community_names) >= 2:
            bridges = agent_tools.find_bridge_people_tool(session, g, community_names[0], community_names[1])
            graph_evidence["bridge_people"] = bridges
            candidate_people += [b["person_id"] for b in bridges]
        elif len(community_names) == 1:
            analysis = agent_tools.analyze_community(session, g, community_names[0])
            graph_evidence["community_analysis"] = analysis
            candidate_people += analysis.get("top_hubs_by_degree", [])
        else:
            # "what communities am I connected to" — look up the user's own memberships
            profile = agent_tools.get_person_profile_tool(session, user_id)
            graph_evidence["user_communities"] = profile["communities"] if profile else []

    if PATH_ANALYSIS in active_nodes:
        log.append("find_shortest_path / find_connections")
        targets = [p for p in state.get("candidate_people", []) if p != user_id]
        paths = {}
        for target in targets:
            paths[target] = agent_tools.find_shortest_path(g, user_id, target)
        graph_evidence["paths"] = paths
        if intent == "outside_network" and semantic_evidence.get("semantic_hits"):
            outside_ids = agent_tools.find_relevant_outside_network(
                g, user_id, semantic_evidence["semantic_hits"], max_known_degree=1
            )
            graph_evidence["outside_network_ids"] = outside_ids
            candidate_people += outside_ids

    if TEMPORAL in active_nodes:
        log.append("get_temporal_relationships")
        recent = agent_tools.get_temporal_relationships(session, user_id, days=constraints.get("days", 30))
        graph_evidence["recent_relationships"] = recent
        candidate_people += [r["id"] for r in recent]

    candidate_people = list(dict.fromkeys(p for p in candidate_people if p != user_id))

    return {
        "graph_evidence": graph_evidence,
        "semantic_evidence": semantic_evidence,
        "candidate_people": candidate_people,
        "tool_calls_log": log,
    }


def fuse_evidence_node(state: AgentState) -> dict:
    session = state["_session"]
    g = state["_graph"]
    user_id = state["user_id"]
    entities = state["entities"]
    query_text = entities.get("goal_text") or state["user_query"]

    # Only run semantic retrieval here if the query's intent actually calls for it
    # (spec §10: don't force every query through every tool, including at fusion time).
    use_semantic = SEMANTIC_RETRIEVAL in tool_nodes_for_intent(state["intent"])

    pool = build_candidate_pool(
        session, g, user_id,
        graph_candidate_ids=state.get("candidate_people", []),
        semantic_query=query_text if use_semantic else None,
        semantic_top_k=20,
    )

    # attach outside-network filter if requested
    if state["intent"] == "outside_network":
        max_degree = state["constraints"].get("max_degree") or 1
        pool = {pid: c for pid, c in pool.items() if c.network_degree is None or c.network_degree > max_degree}

    # batch-resolve every person id that appears in any candidate's path, so path
    # descriptions can show real names instead of raw ids like "p_0001 -> p_0165"
    path_ids = {pid for cand in pool.values() if cand.path for pid in cand.path}
    path_names = get_names_for_ids(session, list(path_ids)) if path_ids else {}
    path_names[user_id] = "you"

    evidence_bundles: dict[str, EvidenceBundle] = {}
    for pid, cand in pool.items():
        bundle = EvidenceBundle(person_id=pid)
        if cand.network_degree is not None:
            bundle.metrics.append(CalculatedMetric(
                subject_id=pid, metric_name="network_degree", value=cand.network_degree,
                description=f"{cand.name} is a {cand.network_degree}-degree connection.",
            ))
        if cand.path:
            readable_path = " -> ".join(path_names.get(p, p) for p in cand.path)
            bundle.facts.append(DatabaseFact(
                subject_id=pid, fact_type="path",
                description=f"Path: {readable_path}",
                data={"path": cand.path, "path_names": [path_names.get(p, p) for p in cand.path]},
            ))
        if cand.relationship_strength:
            bundle.facts.append(DatabaseFact(
                subject_id=pid, fact_type="relationship_strength",
                description=f"Direct relationship strength with {cand.name}: {cand.relationship_strength:.2f}.",
                data={"strength": cand.relationship_strength},
            ))
        for interest in cand.shared_interests:
            bundle.facts.append(DatabaseFact(
                subject_id=pid, fact_type="shared_interest",
                description=f"Shares interest in {interest} with you.",
                data={"interest": interest},
            ))
        for comm in cand.shared_communities:
            bundle.facts.append(DatabaseFact(
                subject_id=pid, fact_type="shared_community",
                description=f"Member of {comm}, same as you.",
                data={"community": comm},
            ))
        if cand.semantic_score:
            bundle.semantic.append(SemanticScore(
                subject_id=pid, query_text=query_text, score=cand.semantic_score,
                description=f"Semantic relevance to your query: {cand.semantic_score:.2f}.",
            ))
        evidence_bundles[pid] = bundle

    return {
        "graph_evidence": {**state.get("graph_evidence", {}), "_candidate_pool": pool,
                           "_evidence_bundles": evidence_bundles},
    }


def rank_candidates_node(state: AgentState) -> dict:
    session = state["_session"]
    user_id = state["user_id"]
    entities = state["entities"]
    pool = state["graph_evidence"]["_candidate_pool"]

    ctx = QueryContext(goal_text=entities.get("goal_text"), location=entities.get("location"))
    features_list = [compute_features(session, user_id, cand, ctx) for cand in pool.values()]
    ranked = score_candidates(features_list, profile=state.get("ranking_profile"))

    top_k = state["constraints"].get("top_k", TOP_K_FOR_EXPLANATION)
    top_ranked = ranked[:top_k]

    ranking_features = {rc.person_id: rc for rc in ranked}

    return {
        "ranking_features": ranking_features,
        "ranked_candidates": [
            {"person_id": rc.person_id, "score": rc.score, "normalized_features": rc.normalized_features,
             "weights_used": rc.weights_used}
            for rc in top_ranked
        ],
    }


def validate_evidence_node(state: AgentState) -> dict:
    """Drops any ranked candidate whose evidence bundle is empty — mechanically
    enforces that every recommendation is evidence-backed (spec §17)."""
    bundles: dict[str, EvidenceBundle] = state["graph_evidence"].get("_evidence_bundles", {})
    validated = []
    for rc in state.get("ranked_candidates", []):
        pid = rc["person_id"]
        bundle = bundles.get(pid)
        if bundle and not bundle.is_empty():
            validated.append(rc)
        else:
            logger.info("Dropping candidate %s: no supporting evidence", pid)
    return {"ranked_candidates": validated}


def generate_explanation_node(state: AgentState) -> dict:
    llm = get_llm()
    bundles: dict[str, EvidenceBundle] = state["graph_evidence"].get("_evidence_bundles", {})
    pool = state["graph_evidence"].get("_candidate_pool", {})

    blocks = []
    for rc in state["ranked_candidates"]:
        pid = rc["person_id"]
        name = pool[pid].name if pid in pool else pid
        bundle = bundles.get(pid)
        lines = bundle.summary_lines() if bundle else []
        blocks.append(f"- {name} (score={rc['score']:.2f}):\n  " + "\n  ".join(lines))
    candidates_block = "\n".join(blocks) if blocks else "(no candidates)"

    messages = [
        {"role": "system", "content": prompts.EXPLANATION_SYSTEM},
        {"role": "user", "content": prompts.EXPLANATION_USER_TEMPLATE.format(
            query=state["user_query"], candidates_block=candidates_block)},
    ]
    response = llm.invoke(messages)
    return {"final_response": response.content}


def community_summary_node(state: AgentState) -> dict:
    """Handles 'what communities am I connected to?' / 'what communities exist?' —
    a direct factual lookup with no person to rank, so it bypasses the
    retrieval/ranking pipeline entirely."""
    session = state["_session"]
    user_id = state["user_id"]

    profile = agent_tools.get_person_profile_tool(session, user_id)
    user_communities = profile["communities"] if profile else []

    all_communities = session.run("MATCH (c:Community) RETURN c.name AS name, c.description AS description")
    all_rows = [dict(r) for r in all_communities]

    if user_communities:
        lines = "\n".join(f"- {c}" for c in user_communities)
        response = f"You are connected to the following communities:\n{lines}"
    else:
        lines = "\n".join(f"- {r['name']}: {r['description']}" for r in all_rows)
        response = f"Here are the communities in the network:\n{lines}"

    return {
        "final_response": response,
        "ranked_candidates": [],
        "graph_evidence": {"user_communities": user_communities, "all_communities": all_rows},
        "tool_calls_log": state.get("tool_calls_log", []) + ["get_person_profile"],
    }


def no_results_node(state: AgentState) -> dict:
    return {"final_response": prompts.NO_RESULTS_TEMPLATE, "ranked_candidates": []}


def why_not_node(state: AgentState) -> dict:
    previous = state.get("previous_ranked_candidates")
    pair = state.get("compare_pair")
    if not previous or not pair:
        return {"final_response": "I don't have a previous ranked answer to compare. Ask a discovery question first."}

    by_id = {rc["person_id"]: rc for rc in previous}
    a_id, b_id = pair
    if a_id not in by_id or b_id not in by_id:
        return {"final_response": "I don't have ranking data for one of those people from the previous answer."}

    from app.ranking.scorer import RankedCandidate
    a = RankedCandidate(person_id=a_id, score=by_id[a_id]["score"],
                         features=None, normalized_features=by_id[a_id]["normalized_features"],
                         weights_used=by_id[a_id].get("weights_used", {}))
    b = RankedCandidate(person_id=b_id, score=by_id[b_id]["score"],
                         features=None, normalized_features=by_id[b_id]["normalized_features"],
                         weights_used=by_id[b_id].get("weights_used", {}))
    explanation = compare_candidates(a, b)

    diff_lines = [
        f"- {d.label}: A={d.a_value:.2f}, B={d.b_value:.2f} (weight={d.weight:.2f}) -> favors {d.advantage.upper()}"
        for d in explanation.diffs
    ]
    llm = get_llm()
    messages = [
        {"role": "system", "content": prompts.WHY_NOT_SYSTEM},
        {"role": "user", "content": prompts.WHY_NOT_USER_TEMPLATE.format(
            query=state["user_query"], a_id=a_id, a_score=a.score, b_id=b_id, b_score=b.score,
            diff_block="\n".join(diff_lines))},
    ]
    response = llm.invoke(messages)
    return {"final_response": response.content}
