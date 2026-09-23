"""Tests for the synthetic data generator and Neo4j bulk loader."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_generator_is_deterministic(tmp_path):
    """Running the generator twice must produce byte-identical output."""
    env_script = REPO_ROOT / "scripts" / "generate_data.py"
    subprocess.run([sys.executable, str(env_script)], check=True, cwd=REPO_ROOT)
    first = (REPO_ROOT / "data" / "generated" / "people.json").read_text()

    subprocess.run([sys.executable, str(env_script)], check=True, cwd=REPO_ROOT)
    second = (REPO_ROOT / "data" / "generated" / "people.json").read_text()

    assert first == second


def test_generated_manifest_counts():
    manifest_path = REPO_ROOT / "data" / "generated" / "manifest.json"
    assert manifest_path.exists(), "run scripts/generate_data.py before testing"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_people"] == 300
    assert manifest["demo_user_id"] == "p_0001"
    assert manifest["demo_bridge_id"] == "p_0002"
    assert "p_0002" in manifest["bridge_people"]
    assert len(manifest["hub_people"]) >= 10


@pytest.mark.integration
def test_neo4j_load_is_idempotent(neo4j_test_session):
    from app.graph.loader import load_all

    stats_1 = load_all(neo4j_test_session, clear=True)
    stats_2 = load_all(neo4j_test_session, clear=False)
    assert stats_1 == stats_2

    count = neo4j_test_session.run("MATCH (p:Person) RETURN count(p) AS c").single()["c"]
    assert count == stats_1["people"]


@pytest.mark.integration
def test_neo4j_load_produces_expected_bridge_person(neo4j_test_session):
    from app.graph.loader import load_all

    load_all(neo4j_test_session, clear=True)
    result = neo4j_test_session.run(
        """
        MATCH (m:Person {id: 'p_0002'})-[:MEMBER_OF]->(c:Community)
        RETURN collect(c.name) AS communities
        """
    ).single()
    assert set(result["communities"]) == {"Contemporary Dance Circle", "Independent Musicians Network"}
