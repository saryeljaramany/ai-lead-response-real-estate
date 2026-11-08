"""
tests/test_schema_contract.py

Schema-contract tests verifying synchronization between:
- db/migrations/001_create_leads.sql
- models/lead.py (LeadRecord, LeadInput, LeadScore)
- services/supabase_service.py (_record_to_row)

Rules enforced:
- No network access; .env is never read.
- main.py is never imported.
"""

from pathlib import Path
import re
import unittest

from models import LeadInput, LeadRecord, LeadScore
from models.lead import MAX_SCORE, MIN_SCORE
import services.supabase_service as svc


class TestSchemaContract(unittest.TestCase):
    """Verify that SQL schema and Pydantic models remain in exact agreement."""

    @classmethod
    def setUpClass(cls):
        cls.sql_path = (
            Path(__file__).resolve().parent.parent
            / "db"
            / "migrations"
            / "001_create_leads.sql"
        )
        if not cls.sql_path.is_file():
            raise FileNotFoundError(
                f"Migration file missing at expected path: {cls.sql_path}"
            )
        cls.sql_content = cls.sql_path.read_text(encoding="utf-8")

    def test_1_parsed_sql_columns_match_record_to_row(self):
        """Parse column names from CREATE TABLE block and compare to _record_to_row."""
        match = re.search(
            r"create table if not exists public\.leads \((.*?)\);",
            self.sql_content,
            re.DOTALL | re.IGNORECASE,
        )
        self.assertIsNotNone(
            match,
            "Could not find 'create table if not exists public.leads' block in SQL",
        )
        table_body = match.group(1)

        sql_columns = []
        for line in table_body.splitlines():
            # Lines starting with two spaces and an identifier, ignoring comments and constraints
            if line.startswith("  "):
                stripped = line.strip()
                if not stripped or stripped.startswith("--"):
                    continue
                first_token = stripped.split()[0]
                if first_token.lower() == "constraint":
                    continue
                sql_columns.append(first_token)

        sample = LeadRecord(
            id="3fa85f64-5717-4562-b3fc-2c963f66afa6",
            lead=LeadInput(
                name="Sarah Mitchell",
                email="sarah@example.com",
                phone="+1-415-555-0192",
                message="Enquiry",
            ),
            score=LeadScore(
                score=80,
                reasoning="Strong lead",
                suggested_action="Call today",
                priority_level="normal",
            ),
        )
        row = svc._record_to_row(sample)

        self.assertEqual(
            set(sql_columns),
            set(row.keys()),
            f"SQL columns {set(sql_columns)} do not match _record_to_row keys {set(row.keys())}",
        )
        self.assertEqual(len(sql_columns), 18)

    def test_2_sql_score_check_matches_model_bounds(self):
        """The SQL contains 'between 0 and 100' matching models.lead MIN_SCORE/MAX_SCORE."""
        expected_range = f"between {MIN_SCORE} and {MAX_SCORE}"
        self.assertIn(
            expected_range,
            self.sql_content,
            f"Expected '{expected_range}' in migration SQL",
        )

    def test_3_lead_score_categories_and_priorities_match_sql(self):
        """LeadScore derived categories and priority levels must agree with SQL constraints."""
        # Extract SQL category constraints: check (category in ('hot', 'warm', 'cold'))
        cat_match = re.search(
            r"category\s+in\s*\(([^)]+)\)",
            self.sql_content,
            re.IGNORECASE,
        )
        self.assertIsNotNone(cat_match, "Could not find category check constraint in SQL")
        sql_categories = {
            c.strip().strip("'\"") for c in cat_match.group(1).split(",")
        }

        # Test boundary and middle scores: 0, 39, 40, 69, 70, 100
        for score_val in [0, 39, 40, 69, 70, 100]:
            with self.subTest(score=score_val):
                score_obj = LeadScore(
                    score=score_val,
                    reasoning="Score test",
                    suggested_action="Action",
                    priority_level="normal",
                )
                self.assertIn(
                    score_obj.category,
                    sql_categories,
                    f"Derived category '{score_obj.category}' for score {score_val} not in SQL categories {sql_categories}",
                )

        # Extract SQL priority constraints: check (priority_level in ('urgent', 'normal', 'low'))
        prio_match = re.search(
            r"priority_level\s+in\s*\(([^)]+)\)",
            self.sql_content,
            re.IGNORECASE,
        )
        self.assertIsNotNone(prio_match, "Could not find priority_level check constraint in SQL")
        sql_priorities = {
            p.strip().strip("'\"") for p in prio_match.group(1).split(",")
        }

        # Every priority in SQL must be accepted by LeadScore
        for prio in sql_priorities:
            with self.subTest(priority=prio):
                score_obj = LeadScore(
                    score=50,
                    reasoning="Priority test",
                    suggested_action="Action",
                    priority_level=prio,
                )
                self.assertEqual(score_obj.priority_level, prio)

        # 'critical' is rejected
        with self.assertRaises(ValueError):
            LeadScore(
                score=50,
                reasoning="Invalid priority",
                suggested_action="Action",
                priority_level="critical",
            )

    def test_4_sql_security_and_grants(self):
        """SQL enables RLS, grants exactly select, insert, update to service_role, and no delete."""
        lower_sql = self.sql_content.lower()

        # 1. Row level security enabled
        self.assertIn(
            "alter table public.leads enable row level security;",
            lower_sql,
            "Migration SQL must enable row level security on public.leads",
        )

        # 2. Grant to service_role is exactly select, insert, update
        grant_match = re.search(
            r"grant\s+(.*?)\s+on\s+table\s+public\.leads\s+to\s+service_role;",
            lower_sql,
        )
        self.assertIsNotNone(
            grant_match,
            "Could not find 'grant ... on table public.leads to service_role' in SQL",
        )
        granted_privs = {
            p.strip() for p in grant_match.group(1).split(",")
        }
        self.assertEqual(
            granted_privs,
            {"select", "insert", "update"},
            f"Expected privileges {{'select', 'insert', 'update'}}, got {granted_privs}",
        )

        # 3. No create policy
        self.assertNotIn(
            "create policy",
            lower_sql,
            "Migration SQL must not define policies (service_role bypasses RLS)",
        )

        # 4. No grant to anon or authenticated
        anon_grants = re.findall(
            r"grant\s+.*?\s+to\s+[^;]*(?:anon|authenticated)",
            lower_sql,
        )
        self.assertEqual(
            anon_grants,
            [],
            f"Found forbidden grants to anon/authenticated: {anon_grants}",
        )

        # 5. No delete grant
        delete_grants = re.findall(
            r"grant\s+[^;]*\bdelete\b[^;]*\s+to\s+",
            lower_sql,
        )
        self.assertEqual(
            delete_grants,
            [],
            f"Found forbidden delete grant in SQL: {delete_grants}",
        )


if __name__ == "__main__":
    unittest.main()
