[README.md](https://github.com/user-attachments/files/32569865/README.md)
# AI Social Graph Agent

A tool-using AI agent that reasons over a rich, general-purpose 300-person synthetic social
graph — spanning AI, dance, music, fashion, beauty, photography, design, travel, food, sports,
entrepreneurship, film, arts, education, and wellness — to answer natural-language questions
about relationships, communities, bridges between communities, and goal-based collaborator
discovery, with every recommendation grounded in retrieved evidence.

## Problem & motivation

Most "recommendation" demos either do pure keyword/graph lookups (miss anything phrased
differently) or pure semantic search (miss structural context like "how close are we,
and through whom"). Real social discovery needs both — plus the ability to explain *why*
someone was recommended, using facts an LLM didn't invent. This project builds that as a
complete, evidence-grounded agent rather than a single prompt.

## Architecture

```
Neo4j (graph store)  ──export──▶  NetworkX (in-memory analytics: centrality, community, paths)
Qdrant (vector store) ◀──embed── sentence-transformers/all-MiniLM-L6-v2
        │                               │
        └────────────┬──────────────────┘
                      ▼
              LangGraph agent (app/agent/)
   understand_query → run_tools → fuse_evidence → rank_candidates
   → validate_evidence → generate_explanation
                      ▼
              Streamlit UI (app/ui/)
```

## Data model

- **Nodes**: `Person`, `Interest` (hierarchical, e.g. `Arts > Dance > Kathak`), `Skill`,
  `Community`, `Project`, `City`.
- **Relationships**: `KNOWS` / `FRIENDS_WITH` / `WORKED_WITH` / `COLLABORATED_WITH` /
  `FOLLOWS` / `MET_AT` (all with `strength`, `since`, `last_interaction`,
  `interaction_count`, `context`), `MEMBER_OF`, `HAS_SKILL`, `INTERESTED_IN`, `WORKED_ON`,
  `LIVES_IN`, `CHILD_OF` (interest hierarchy).
- **Generator** (`scripts/generate_data.py`): deterministic, seed=42. Builds 10
  community archetypes with realistic size variation, deliberately injects ~20 bridge
  people and ~12 network hubs (so bridge-detection and hub queries have reliable,
  demoable answers rather than hoping for incidental structure), and hand-authors one
  primary demo user (`p_0001`, "Aisha Kapoor") whose profile spans AI, dance, fashion, and
  travel photography — plus one named bridge person, "Maya Krishnan" (`p_0002`), who
  explicitly bridges the Contemporary Dance Circle and Independent Musicians Network,
  mirroring the spec's own worked example.

## Agent workflow (LangGraph)

```
understand_query (LLM: intent + entity extraction)
        │
        ├── why_not? ──────────────────────────► why_not_response (uses prior turn's
        │                                          ranking features, no re-retrieval)
        ├── "what communities am I in?" ────────► community_summary_response (direct
        │                                          profile lookup, no ranking needed)
        │
        ▼
   run_tools  (only the tool groups the classified intent needs — see
               app/agent/routing.py's INTENT_TO_TOOL_NODES dispatch table)
        │
        ▼
  fuse_evidence  (merges graph + semantic candidates, attaches evidence objects)
        │
        ▼
  rank_candidates  (config/ranking_weights.yaml, 7 weighted signals)
        │
        ▼
  validate_evidence  (drops any candidate without a supporting evidence object)
        │
        ├── no evidence-backed candidates ─────► no_results_response
        └── otherwise ──────────────────────────► generate_explanation (LLM, evidence-
                                                    constrained prompt)
```

Tool selection is a deterministic table keyed by LLM-classified intent (not LLM
function-calling) — see `docs/PROJECT_JOURNAL.md` for why. This keeps "don't force every
query through every tool" (spec requirement) both correct and independently unit-testable
without mocking the LLM.

### The 11 tools

`search_people`, `semantic_search`, `find_connections`, `find_shortest_path`,
`find_shared_interests`, `find_shared_communities`, `find_bridge_people`,
`get_person_profile`, `analyze_community`, `find_relevant_outside_network`,
`get_temporal_relationships` — implemented in `app/agent/tools.py` as thin adapters over
`app/graph/`, `app/analytics/`, and `app/retrieval/`, so each is independently testable.

## Graph algorithms (`app/analytics/algorithms.py`)

| Algorithm | Product purpose |
|---|---|
| Degree centrality / PageRank | "Who are the network hubs / structurally influential people?" |
| Betweenness centrality (pair-restricted, see below) | "Who bridges my dance and music communities?" |
| Shortest path | "How am I connected to Alex?" |
| Louvain community detection | "What communities exist in the network?" |

**Bridge detection note**: raw global betweenness centrality over-favors generic
high-degree hub nodes (whose many ties are mostly unrelated to either community) over
someone deliberately embedded in *both* specific communities. `find_bridge_people`
instead ranks by a pair-specific `bridge_strength = min(weighted ties into community A,
weighted ties into community B)` — a bottleneck metric — with global betweenness kept as
a secondary signal for explanation. See `docs/PROJECT_JOURNAL.md`.

## Semantic layer & embeddings

Text (bios, interest names, skill names, project/community descriptions) is embedded with
`sentence-transformers/all-MiniLM-L6-v2` (384-dim) and stored in Qdrant, one collection per
entity type. Retrieval scores candidates by **cosine similarity** — the cosine of the angle
between two embedding vectors, in `[-1, 1]`, where 1 means the two texts point in the same
direction in meaning-space. This is what lets a query like *"creative storytelling"*
surface people/interests tied to film, photography, dance, or music even when none of
those exact words appear in the query.

## Hybrid retrieval & ranking

Graph retrieval answers *who is connected, how far, via which communities*. Semantic
retrieval answers *who is meaning-relevant even without exact keyword or graph overlap*.
`app/retrieval/hybrid.py` merges both into one candidate pool; `app/ranking/scorer.py`
combines 7 documented, YAML-configured signals (`config/ranking_weights.yaml`):
`semantic_relevance`, `graph_proximity`, `relationship_strength`, `shared_interests`,
`shared_communities`, `goal_alignment`, `location_relevance`, `recency`. Each is min-max
normalized across the current candidate set before weighting, and ties break
deterministically — so the same query always produces the same ranking.

## Evidence grounding

Every claim in a response traces back to one of four explicit evidence tiers
(`app/agent/evidence.py`): `DatabaseFact`, `CalculatedMetric`, `SemanticScore`, or
`LLMInterpretation`. `validate_evidence` mechanically drops any ranked candidate with no
supporting `DatabaseFact` / `CalculatedMetric` / `SemanticScore` before the LLM ever sees
them — the LLM is only ever shown evidence-backed candidates and is explicitly instructed
not to introduce facts beyond what's provided.

## Evaluation

`app/evaluation/run_eval.py` runs ~30 benchmark queries (`app/evaluation/
benchmark_queries.py`) across 9 categories — relationship, community, bridge detection,
goal-based discovery, outside-network discovery, semantic similarity, location, temporal,
cross-domain — through **the same agent and scorer code path** under three ranking
profiles (`semantic_only`, `graph_only`, `hybrid`), so the comparison is apples-to-apples.
Gold answers are computed live from the graph/DB (not hand-typed IDs) so the benchmark
stays correct even if the generator changes. Metrics: Precision@5, Recall@5, MRR,
graph-path correctness, evidence groundedness, hallucination rate, latency. Results are
written to [`docs/EVALUATION_RESULTS.md`](docs/EVALUATION_RESULTS.md).

```bash
python -m app.evaluation.run_eval
```

## Setup

**Prerequisites**: Python 3.11+, Docker Desktop, an OpenAI API key.

```bash
# 1. Clone and enter the repo, then create a virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# 3. Start Neo4j + Qdrant
docker compose up -d

# 4. Generate data, load the graph, build embeddings
python scripts/seed_all.py

# 5. Run tests
pytest

# 6. Try the CLI
python main.py "Who could help me with an AI + fashion project?"

# 7. Or launch the UI
streamlit run app/ui/streamlit_app.py
```

## Usage examples

```bash
python main.py "How am I connected to Maya Krishnan?"
python main.py "Who connects my dance and music communities?"
python main.py "What communities am I connected to?"
python main.py "Who interested in music is near Bangalore?"
python main.py "Who have I recently become connected to?"
```

## Tests

```bash
pytest                    # full suite (needs Neo4j + Qdrant running)
pytest -m "not integration"  # fast, pure-logic subset only
```

Covers: graph loading (idempotency, determinism), graph queries, shortest paths,
community/bridge detection, semantic retrieval, ranking math and tie-break determinism,
agent routing (pure logic, no LLM needed), evidence validation, and edge cases (empty/
invalid queries, missing people, simulated Neo4j/OpenAI failures).

## Limitations

- Semantic quality over person **bios** is template-generated text, so cosine scores on
  bios alone are moderate (~0.4-0.6); the hybrid design compensates by weighting graph
  signals (shared interests/communities, network proximity) alongside semantic score
  rather than relying on bio embeddings in isolation.
- Location filtering is a ranking signal, not a hard filter, in most intents — a
  deliberate design choice (see `docs/PROJECT_JOURNAL.md`) so location-adjacent
  candidates aren't excluded outright, but it means "near Bangalore" queries can surface
  a few non-Bangalore candidates lower in the ranking.
- Evidence groundedness/hallucination-rate metrics are pragmatic keyword-overlap checks,
  not a full NLI/entailment model — adequate for this project's scope but not a
  production-grade grounding verifier.
- Single-user demo: there is one hand-authored primary user (`p_0001`); the system is not
  multi-tenant (by design — see "Out of scope" in the spec).
