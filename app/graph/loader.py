"""Bulk-load generated JSON data into Neo4j. Idempotent via MERGE."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import DATA_GENERATED_DIR
from app.graph.schema import apply_schema
from app.logging_config import get_logger

logger = get_logger(__name__)


def _load_json(name: str) -> object:
    path = Path(DATA_GENERATED_DIR) / name
    with open(path) as f:
        return json.load(f)


def clear_graph(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def load_people(session, people: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (p:Person {id: row.id})
        SET p.name = row.name, p.age_range = row.age_range, p.bio = row.bio,
            p.role = row.role, p.city = row.city, p.country = row.country,
            p.created_at = row.created_at
        """,
        rows=people,
    )


def load_interests(session, interests: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (i:Interest {id: row.id})
        SET i.name = row.name, i.level = row.level
        """,
        rows=interests,
    )
    session.run(
        """
        UNWIND $rows AS row
        MATCH (child:Interest {id: row.id})
        MATCH (parent:Interest {id: row.parent_id})
        MERGE (child)-[:CHILD_OF]->(parent)
        """,
        rows=[r for r in interests if r["parent_id"]],
    )


def load_skills(session, skills: list[dict]) -> None:
    session.run(
        "UNWIND $rows AS row MERGE (s:Skill {id: row.id}) SET s.name = row.name",
        rows=skills,
    )


def load_cities(session, cities: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:City {id: row.id})
        SET c.name = row.name, c.country = row.country, c.lat = row.lat, c.lng = row.lng
        """,
        rows=cities,
    )


def load_communities(session, communities: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (c:Community {id: row.id})
        SET c.name = row.name, c.description = row.description
        """,
        rows=communities,
    )


def load_projects(session, projects: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MERGE (pr:Project {id: row.id})
        SET pr.name = row.name, pr.description = row.description,
            pr.start_date = row.start_date, pr.end_date = row.end_date
        """,
        rows=projects,
    )


def load_person_person(session, edges: list[dict]) -> None:
    by_type: dict[str, list[dict]] = {}
    for e in edges:
        by_type.setdefault(e["type"], []).append(e)
    for rel_type, rows in by_type.items():
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (a:Person {{id: row.from}})
            MATCH (b:Person {{id: row.to}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.strength = row.strength, r.since = row.since,
                r.last_interaction = row.last_interaction,
                r.interaction_count = row.interaction_count, r.context = row.context
            """,
            rows=rows,
        )


def load_member_of(session, rows: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {id: row.person})
        MATCH (c:Community {id: row.community})
        MERGE (p)-[r:MEMBER_OF]->(c)
        SET r.joined_at = row.joined_at, r.role = row.role
        """,
        rows=rows,
    )


def load_has_skill(session, rows: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {id: row.person})
        MATCH (s:Skill {id: row.skill})
        MERGE (p)-[r:HAS_SKILL]->(s)
        SET r.proficiency = row.proficiency
        """,
        rows=rows,
    )


def load_interested_in(session, rows: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {id: row.person})
        MATCH (i:Interest {id: row.interest})
        MERGE (p)-[r:INTERESTED_IN]->(i)
        SET r.strength = row.strength, r.since = row.since
        """,
        rows=rows,
    )


def load_worked_on(session, rows: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {id: row.person})
        MATCH (pr:Project {id: row.project})
        MERGE (p)-[r:WORKED_ON]->(pr)
        SET r.role = row.role, r.since = row.since, r.until = row.until
        """,
        rows=rows,
    )


def load_lives_in(session, rows: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MATCH (p:Person {id: row.person})
        MATCH (c:City {id: row.city})
        MERGE (p)-[:LIVES_IN]->(c)
        """,
        rows=rows,
    )


def load_all(session, clear: bool = True) -> dict:
    if clear:
        logger.info("Clearing existing graph")
        clear_graph(session)

    apply_schema(session)

    people = _load_json("people.json")
    interests = _load_json("interests.json")
    skills = _load_json("skills.json")
    cities = _load_json("cities.json")
    communities = _load_json("communities.json")
    projects = _load_json("projects.json")
    relationships = _load_json("relationships.json")

    logger.info("Loading %d people", len(people))
    load_people(session, people)
    logger.info("Loading %d interests", len(interests))
    load_interests(session, interests)
    logger.info("Loading %d skills", len(skills))
    load_skills(session, skills)
    logger.info("Loading %d cities", len(cities))
    load_cities(session, cities)
    logger.info("Loading %d communities", len(communities))
    load_communities(session, communities)
    logger.info("Loading %d projects", len(projects))
    load_projects(session, projects)

    logger.info("Loading %d person-person edges", len(relationships["person_person"]))
    load_person_person(session, relationships["person_person"])
    logger.info("Loading %d MEMBER_OF edges", len(relationships["member_of"]))
    load_member_of(session, relationships["member_of"])
    logger.info("Loading %d HAS_SKILL edges", len(relationships["has_skill"]))
    load_has_skill(session, relationships["has_skill"])
    logger.info("Loading %d INTERESTED_IN edges", len(relationships["interested_in"]))
    load_interested_in(session, relationships["interested_in"])
    logger.info("Loading %d WORKED_ON edges", len(relationships["worked_on"]))
    load_worked_on(session, relationships["worked_on"])
    logger.info("Loading %d LIVES_IN edges", len(relationships["lives_in"]))
    load_lives_in(session, relationships["lives_in"])

    return {
        "people": len(people),
        "interests": len(interests),
        "skills": len(skills),
        "cities": len(cities),
        "communities": len(communities),
        "projects": len(projects),
        "person_person_edges": len(relationships["person_person"]),
    }
