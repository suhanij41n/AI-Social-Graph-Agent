"""Node labels, relationship types, and Cypher DDL for constraints/indexes."""

# Node labels
PERSON = "Person"
INTEREST = "Interest"
SKILL = "Skill"
COMMUNITY = "Community"
PROJECT = "Project"
CITY = "City"

# Relationship types between people
PERSON_REL_TYPES = [
    "KNOWS",
    "FRIENDS_WITH",
    "WORKED_WITH",
    "COLLABORATED_WITH",
    "FOLLOWS",
    "MET_AT",
]

# Relationship types to other entities
MEMBER_OF = "MEMBER_OF"          # Person -> Community
HAS_SKILL = "HAS_SKILL"          # Person -> Skill
INTERESTED_IN = "INTERESTED_IN"  # Person -> Interest
WORKED_ON = "WORKED_ON"          # Person -> Project
LIVES_IN = "LIVES_IN"            # Person -> City
CHILD_OF = "CHILD_OF"            # Interest -> Interest (hierarchy)
RELATED_TO = "RELATED_TO"        # Interest -> Interest (lateral, optional)

CONSTRAINTS = [
    f"CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:{PERSON}) REQUIRE p.id IS UNIQUE",
    f"CREATE CONSTRAINT interest_id IF NOT EXISTS FOR (i:{INTEREST}) REQUIRE i.id IS UNIQUE",
    f"CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:{SKILL}) REQUIRE s.id IS UNIQUE",
    f"CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:{COMMUNITY}) REQUIRE c.id IS UNIQUE",
    f"CREATE CONSTRAINT project_id IF NOT EXISTS FOR (pr:{PROJECT}) REQUIRE pr.id IS UNIQUE",
    f"CREATE CONSTRAINT city_id IF NOT EXISTS FOR (ci:{CITY}) REQUIRE ci.id IS UNIQUE",
]

INDEXES = [
    f"CREATE INDEX person_city IF NOT EXISTS FOR (p:{PERSON}) ON (p.city)",
    f"CREATE INDEX person_name IF NOT EXISTS FOR (p:{PERSON}) ON (p.name)",
]


def apply_schema(session) -> None:
    for stmt in CONSTRAINTS + INDEXES:
        session.run(stmt)
