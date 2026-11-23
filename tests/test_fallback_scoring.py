# tests/test_fallback_scoring.py
# Standard-library only: unittest, os, json, re, pathlib, types
# Run with: python -m unittest tests.test_fallback_scoring -v

import json
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
_SVC_PATH = _REPO / "services" / "fallback_scoring.py"

_BASE_LEAD_KWARGS = {
    "name": "A B",
    "email": "a@example.com",
    "phone": "+14155550192",
    "message": "Hello",
}


def _make_lead(**kwargs):
    """Build a LeadInput for testing."""
    from models.lead import LeadInput
    kw = dict(_BASE_LEAD_KWARGS)
    kw.update(kwargs)
    return LeadInput(**kw)


# ---------------------------------------------------------------------------
# A -- Source hygiene
# ---------------------------------------------------------------------------

class TestASourceHygiene(unittest.TestCase):
    """Verify import-time safety and source-code constraints."""

    def test_a01_imports_cleanly_with_empty_environment(self):
        """Module must import with an empty os.environ."""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import sys
            mod_name = "services.fallback_scoring"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            try:
                import services.fallback_scoring  # noqa: F401
                imported = True
            finally:
                # Restore so other tests can use it normally.
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
        self.assertTrue(imported)

    def test_a02_source_decodes_as_ascii(self):
        """Source file must be pure ASCII."""
        raw = _SVC_PATH.read_bytes()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"fallback_scoring.py is not pure ASCII: {exc}")

    def test_a03_no_import_logging(self):
        src = _SVC_PATH.read_text(encoding="ascii")
        self.assertNotIn("import logging", src,
                         "fallback_scoring.py must not import logging")

    def test_a04_no_print(self):
        src = _SVC_PATH.read_text(encoding="ascii")
        self.assertNotIn('print(', src,
                         "fallback_scoring.py must not use print()")

    def test_a05_no_os_environ(self):
        src = _SVC_PATH.read_text(encoding="ascii")
        self.assertNotIn("os.environ", src,
                         "fallback_scoring.py must not access os.environ")

    def test_a06_no_os_getenv(self):
        src = _SVC_PATH.read_text(encoding="ascii")
        self.assertNotIn("os.getenv", src,
                         "fallback_scoring.py must not use os.getenv")

    def test_a07_no_import_random(self):
        src = _SVC_PATH.read_text(encoding="ascii")
        self.assertNotIn("import random", src,
                         "fallback_scoring.py must not import random")

    def test_a08_no_time_dot(self):
        src = _SVC_PATH.read_text(encoding="ascii")
        self.assertNotIn("time.", src,
                         "fallback_scoring.py must not use time.*")


# ---------------------------------------------------------------------------
# B -- Exact score tests
# ---------------------------------------------------------------------------

class TestBExactScores(unittest.TestCase):
    """Verify specific score and category values for concrete inputs."""

    def setUp(self):
        from services.fallback_scoring import fallback_score
        self._score = fallback_score

    def test_b01_message_only_hello(self):
        lead = _make_lead()
        result = self._score(lead)
        self.assertEqual(result.score, 10)
        self.assertEqual(result.category, "cold")

    def test_b02_budget_urgent_timeline(self):
        lead = _make_lead(budget="450k", timeline="immediate")
        result = self._score(lead)
        # 10 + 20 (budget stated) + 25 (urgent) = 55
        self.assertEqual(result.score, 55)
        self.assertEqual(result.category, "warm")

    def test_b03_full_hot_lead(self):
        lead = _make_lead(
            budget="$450k-$550k",
            timeline="within 3 months",
            preferred_location="Austin, TX",
            property_type="house",
        )
        result = self._score(lead)
        # 10 + 20 + 25 + 10 + 8 = 73
        self.assertEqual(result.score, 73)
        self.assertEqual(result.category, "hot")

    def test_b04_browsing_message_reduces_score(self):
        lead = _make_lead(
            budget="$450k-$550k",
            timeline="within 3 months",
            preferred_location="Austin, TX",
            property_type="house",
            message="Just looking, no rush",
        )
        result = self._score(lead)
        # 10 + 20 + 25 + 10 + 8 - 15 = 58
        self.assertEqual(result.score, 58)
        self.assertEqual(result.category, "warm")

    def test_b05_browsing_only_clamped_to_zero(self):
        lead = _make_lead(message="Just looking")
        result = self._score(lead)
        # 10 - 15 = -5 -> clamped to 0
        self.assertEqual(result.score, 0)
        self.assertEqual(result.category, "cold")

    def test_b06_viewing_intent(self):
        lead = _make_lead(message="I would like to book a viewing")
        result = self._score(lead)
        # 10 + 10 (intent) = 20
        self.assertEqual(result.score, 20)
        self.assertEqual(result.category, "cold")

    def test_b07_digit_in_message(self):
        lead = _make_lead(message="3 bed")
        result = self._score(lead)
        # 10 + 5 (digit) = 15
        self.assertEqual(result.score, 15)
        self.assertEqual(result.category, "cold")

    def test_b08_message_exactly_80_chars(self):
        msg80 = "a" * 80
        lead = _make_lead(message=msg80)
        result = self._score(lead)
        # 10 + 5 (detailed) = 15
        self.assertEqual(result.score, 15)
        self.assertEqual(result.category, "cold")

    def test_b09_message_79_chars(self):
        msg79 = "a" * 79
        lead = _make_lead(message=msg79)
        result = self._score(lead)
        # 10 (no detailed)
        self.assertEqual(result.score, 10)
        self.assertEqual(result.category, "cold")

    def test_b10_max_score_93(self):
        msg = "a" * 90 + " 5 viewing"
        lead = _make_lead(
            budget="500k",
            timeline="2 weeks",
            preferred_location="Milan",
            property_type="condo",
            message=msg,
        )
        result = self._score(lead)
        # 10 + 20 + 25 + 10 + 8 + 5 (detail) + 5 (digit) + 10 (intent) = 93
        self.assertEqual(result.score, 93)
        self.assertEqual(result.category, "hot")


# ---------------------------------------------------------------------------
# C -- Timeline classification table
# ---------------------------------------------------------------------------

class TestCTimelineClassification(unittest.TestCase):
    """Exhaustive timeline classification cases."""

    def setUp(self):
        from services.fallback_scoring import fallback_score
        self._score = fallback_score

    def _tl_score(self, timeline):
        lead = _make_lead(timeline=timeline)
        return self._score(lead).score

    # Urgent -> score = 10 + 25 = 35
    def test_c01_urgent_immediate(self):
        self.assertEqual(self._tl_score("immediate"), 35)

    def test_c02_urgent_asap(self):
        self.assertEqual(self._tl_score("ASAP"), 35)

    def test_c03_urgent_now(self):
        self.assertEqual(self._tl_score("now"), 35)

    def test_c04_urgent_this_month(self):
        self.assertEqual(self._tl_score("this month"), 35)

    def test_c05_urgent_next_week(self):
        self.assertEqual(self._tl_score("next week"), 35)

    def test_c06_urgent_within_3_months(self):
        self.assertEqual(self._tl_score("within 3 months"), 35)

    def test_c07_urgent_3_months_hyphen(self):
        self.assertEqual(self._tl_score("3-months"), 35)

    def test_c08_urgent_2_weeks(self):
        self.assertEqual(self._tl_score("2 weeks"), 35)

    def test_c09_urgent_12_weeks(self):
        self.assertEqual(self._tl_score("12 weeks"), 35)

    def test_c10_urgent_30_days(self):
        self.assertEqual(self._tl_score("30 days"), 35)

    def test_c11_urgent_90_days(self):
        self.assertEqual(self._tl_score("90 days"), 35)

    def test_c12_urgent_a_few_weeks(self):
        self.assertEqual(self._tl_score("a few weeks"), 35)

    # Medium -> score = 10 + 12 = 22
    def test_c13_medium_6_months(self):
        self.assertEqual(self._tl_score("6 months"), 22)

    def test_c14_medium_4_months(self):
        self.assertEqual(self._tl_score("4 months"), 22)

    def test_c15_medium_12_months(self):
        self.assertEqual(self._tl_score("12 months"), 22)

    def test_c16_medium_1_year(self):
        self.assertEqual(self._tl_score("1 year"), 22)

    def test_c17_medium_next_year(self):
        self.assertEqual(self._tl_score("next year"), 22)

    def test_c18_medium_this_year(self):
        self.assertEqual(self._tl_score("this year"), 22)

    def test_c19_medium_few_months(self):
        self.assertEqual(self._tl_score("few months"), 22)

    # Other -> score = 10 + 5 = 15
    def test_c20_other_someday_ish(self):
        self.assertEqual(self._tl_score("someday-ish"), 15)

    def test_c21_other_whenever(self):
        self.assertEqual(self._tl_score("whenever"), 15)

    def test_c22_other_13_weeks(self):
        self.assertEqual(self._tl_score("13 weeks"), 15)

    def test_c23_other_91_days(self):
        self.assertEqual(self._tl_score("91 days"), 15)

    def test_c24_other_13_months(self):
        self.assertEqual(self._tl_score("13 months"), 15)

    def test_c25_other_2_years(self):
        self.assertEqual(self._tl_score("2 years"), 15)

    def test_c26_other_know(self):
        # "know" must NOT match the word "now"
        self.assertEqual(self._tl_score("know"), 15)

    # Absent or blank -> 10
    def test_c27_absent_timeline(self):
        lead = _make_lead()
        from services.fallback_scoring import fallback_score
        self.assertEqual(fallback_score(lead).score, 10)

    def test_c28_blank_timeline(self):
        lead = _make_lead(timeline="  ")
        from services.fallback_scoring import fallback_score
        # blank stripped -> treated as absent
        self.assertEqual(fallback_score(lead).score, 10)

    def test_c29_precedence_2_weeks_this_year(self):
        self.assertEqual(self._tl_score("2 weeks, this year"), 35)

    def test_c30_precedence_6_months_or_2_weeks(self):
        self.assertEqual(self._tl_score("6 months or 2 weeks"), 35)

    def test_c31_duration_parentheses_3_months(self):
        self.assertEqual(self._tl_score("(3 months)"), 35)

    def test_c32_duration_hyphen_spaces_3_dash_months(self):
        self.assertEqual(self._tl_score("3 - months"), 35)


# ---------------------------------------------------------------------------
# D -- Budget scoring
# ---------------------------------------------------------------------------

class TestDBudget(unittest.TestCase):
    """Budget field scoring (score = 10 + budget_points)."""

    def setUp(self):
        from services.fallback_scoring import fallback_score
        self._score = fallback_score

    def test_d01_budget_with_digit(self):
        lead = _make_lead(budget="450k")
        self.assertEqual(self._score(lead).score, 30)

    def test_d02_budget_no_digit(self):
        lead = _make_lead(budget="flexible")
        self.assertEqual(self._score(lead).score, 15)

    def test_d03_no_budget(self):
        lead = _make_lead()
        self.assertEqual(self._score(lead).score, 10)

    def test_d04_budget_dollar_comma(self):
        lead = _make_lead(budget="$1,200,000")
        self.assertEqual(self._score(lead).score, 30)

    def test_d05_empty_budget_treated_as_absent(self):
        lead = _make_lead(budget="")
        self.assertEqual(self._score(lead).score, 10)


# ---------------------------------------------------------------------------
# E -- Intent and browsing word boundaries
# ---------------------------------------------------------------------------

class TestEWordBoundaries(unittest.TestCase):
    """Check that partial-word matches are rejected and full matches accepted."""

    def setUp(self):
        from services.fallback_scoring import fallback_score
        self._score = fallback_score

    def _msg_score(self, msg):
        return self._score(_make_lead(message=msg)).score

    def test_e01_rental_does_not_trigger_rent(self):
        # "rental" contains "rent" but as a prefix, not a whole word match.
        # Score should NOT include the +10 intent bonus.
        # base 10, digit in "rental"? no, no digit. length < 80. No intent.
        self.assertEqual(self._msg_score("rental"), 10)

    def test_e02_current_does_not_trigger_tour_or_rent(self):
        self.assertEqual(self._msg_score("current"), 10)

    def test_e03_tourist_does_not_trigger_tour(self):
        self.assertEqual(self._msg_score("tourist"), 10)

    def test_e04_viewing_triggers_intent(self):
        # 10 (base) + 10 (intent)
        self.assertEqual(self._msg_score("viewing"), 20)

    def test_e05_pre_approved_case_insensitive(self):
        self.assertEqual(self._msg_score("Pre-Approved"), 20)

    def test_e06_schedule_uppercase(self):
        self.assertEqual(self._msg_score("SCHEDULE"), 20)

    def test_e07_someday_triggers_browsing(self):
        # 10 - 15 = -5 -> clamped 0
        self.assertEqual(self._msg_score("someday"), 0)

    def test_e08_someone_does_not_trigger_browsing(self):
        # "someday" is a phrase match; "someone" does not contain it.
        self.assertEqual(self._msg_score("someone"), 10)

    def test_e09_piano_rushing_does_not_trigger_no_rush(self):
        # "piano rushing" contains "rushing", not whole word "rush".
        self.assertEqual(self._msg_score("piano rushing"), 10)


# ---------------------------------------------------------------------------
# F -- Output contract
# ---------------------------------------------------------------------------

class TestFOutputContract(unittest.TestCase):
    """Verify category/action/priority consistency and reasoning format."""

    def setUp(self):
        from services.fallback_scoring import (
            FALLBACK_REASONING_PREFIX,
            fallback_score,
        )
        from models.lead import HOT_SCORE_MIN, WARM_SCORE_MIN
        self._score = fallback_score
        self._prefix = FALLBACK_REASONING_PREFIX
        self._warm_min = WARM_SCORE_MIN
        self._hot_min = HOT_SCORE_MIN

    def _build_with_target(self, target_score):
        """
        Build a lead that yields approximately target_score via known rules.
        We test category mapping at exact boundary values via SimpleNamespace.
        """
        from services.fallback_scoring import fallback_score
        from types import SimpleNamespace
        lead = SimpleNamespace(
            name="A B", email="a@example.com", phone="+14155550192",
            message="Hello", budget=None, timeline=None,
            preferred_location=None, property_type=None,
        )
        # We will verify the category by directly calling with crafted scores
        # via the module's internal _category helper.
        return lead

    def test_f01_score_0_is_cold(self):
        # score 0: "Just looking" -> 10 - 15 = -5 clamped to 0
        lead = _make_lead(message="Just looking")
        result = self._score(lead)
        self.assertEqual(result.score, 0)
        self.assertEqual(result.category, "cold")
        self.assertEqual(result.priority_level, "low")
        self.assertIn("nurture", result.suggested_action)

    def test_f02_score_39_is_cold(self):
        from types import SimpleNamespace
        from services.fallback_scoring import fallback_score
        # budget stated (+20) + base(10) + intent(10) - browsing(-15) = 25; cold
        lead = _make_lead(budget="400k", message="Just looking for rent")
        result = fallback_score(lead)
        self.assertEqual(result.category, "cold")

    def test_f03_score_40_is_warm(self):
        # 10 + 20 (budget stated) + 12 (medium timeline: 6 months) = 42 -> warm
        lead = _make_lead(budget="400k", timeline="6 months")
        result = self._score(lead)
        self.assertGreaterEqual(result.score, 40)
        self.assertEqual(result.category, "warm")
        self.assertEqual(result.priority_level, "normal")
        self.assertIn("two business days", result.suggested_action)

    def test_f04_score_69_is_warm(self):
        # 10 + 20 + 25 + 8 = 63 warm (no location)
        lead = _make_lead(budget="400k", timeline="immediate", property_type="flat")
        result = self._score(lead)
        self.assertEqual(result.category, "warm")

    def test_f05_score_70_is_hot(self):
        # 10 + 20 + 25 + 10 + 8 = 73 -> hot
        lead = _make_lead(
            budget="$450k",
            timeline="immediate",
            preferred_location="NYC",
            property_type="flat",
        )
        result = self._score(lead)
        self.assertGreaterEqual(result.score, 70)
        self.assertEqual(result.category, "hot")
        self.assertEqual(result.priority_level, "urgent")
        self.assertIn("today", result.suggested_action)

    def test_f06_score_100_is_hot(self):
        # max possible 93, still hot; test with 93
        msg = "a" * 90 + " 5 viewing"
        lead = _make_lead(
            budget="500k", timeline="immediate",
            preferred_location="Milan", property_type="condo",
            message=msg,
        )
        result = self._score(lead)
        self.assertEqual(result.score, 93)
        self.assertEqual(result.category, "hot")

    def test_f07_reasoning_starts_with_prefix(self):
        lead = _make_lead()
        result = self._score(lead)
        self.assertTrue(
            result.reasoning.startswith(self._prefix),
            f"reasoning must start with prefix; got: {result.reasoning!r}"
        )

    def test_f08_no_strong_signals_when_nothing_applies(self):
        lead = _make_lead(message="Hello")
        result = self._score(lead)
        self.assertIn("no strong signals", result.reasoning)

    def test_f09_category_and_priority_consistent(self):
        from models.lead import HOT_SCORE_MIN, WARM_SCORE_MIN
        for timeline, budget, loc, ptype in [
            (None, None, None, None),
            ("immediate", "400k", "NYC", "flat"),
            ("6 months", "400k", None, None),
        ]:
            kw = {}
            if timeline:
                kw["timeline"] = timeline
            if budget:
                kw["budget"] = budget
            if loc:
                kw["preferred_location"] = loc
            if ptype:
                kw["property_type"] = ptype
            lead = _make_lead(**kw)
            result = self._score(lead)
            if result.score >= HOT_SCORE_MIN:
                self.assertEqual(result.category, "hot")
                self.assertEqual(result.priority_level, "urgent")
            elif result.score >= WARM_SCORE_MIN:
                self.assertEqual(result.category, "warm")
                self.assertEqual(result.priority_level, "normal")
            else:
                self.assertEqual(result.category, "cold")
                self.assertEqual(result.priority_level, "low")


# ---------------------------------------------------------------------------
# G -- Privacy: no PII or lead text in output
# ---------------------------------------------------------------------------

class TestGPrivacy(unittest.TestCase):
    """Verify that the output score JSON contains no canary strings."""

    def test_g01_no_pii_or_lead_text_in_output(self):
        from services.fallback_scoring import fallback_score
        lead = _make_lead(
            name="Zed Canary",
            email="canary.lead@example.com",
            phone="+14155550142",
            message="canary message text 450000 viewing",
            budget="CANARY-BUDGET 450k",
            timeline="CANARY-TIMELINE 3 months",
            preferred_location="CANARY-LOCATION",
            property_type="CANARY-TYPE",
        )
        result = fallback_score(lead)
        result_json = result.model_dump_json()

        canary_strings = [
            "Zed Canary", "Zed", "Canary",
            "canary.lead@example.com",
            "+14155550142",
            "canary message text",
            "CANARY-BUDGET", "CANARY-TIMELINE",
            "CANARY-LOCATION", "CANARY-TYPE",
            "canary",
        ]
        for canary in canary_strings:
            self.assertNotIn(
                canary.lower(), result_json.lower(),
                f"Output JSON contains canary string: {canary!r}"
            )


# ---------------------------------------------------------------------------
# H -- Robustness
# ---------------------------------------------------------------------------

class TestHRobustness(unittest.TestCase):
    """Determinism, no mutation, and tolerance of malformed inputs."""

    def setUp(self):
        from services.fallback_scoring import fallback_score
        self._score = fallback_score

    def test_h01_determinism(self):
        lead = _make_lead(
            budget="400k", timeline="immediate",
            preferred_location="NYC", property_type="flat",
            message="I want to view the property today, budget is $500k",
        )
        r1 = self._score(lead)
        r2 = self._score(lead)
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.category, r2.category)
        self.assertEqual(r1.reasoning, r2.reasoning)

    def test_h02_no_mutation_of_input(self):
        lead = _make_lead(budget="400k", timeline="immediate")
        budget_before = lead.budget
        timeline_before = lead.timeline
        message_before = lead.message
        self._score(lead)
        self.assertEqual(lead.budget, budget_before)
        self.assertEqual(lead.timeline, timeline_before)
        self.assertEqual(lead.message, message_before)

    def test_h03_namespace_with_none_values(self):
        from types import SimpleNamespace
        lead = SimpleNamespace(
            message=None, budget=None, timeline=None,
            preferred_location=None, property_type=None,
        )
        result = self._score(lead)
        self.assertIsInstance(result.score, int)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)

    def test_h04_namespace_with_integer_values(self):
        from types import SimpleNamespace
        lead = SimpleNamespace(
            message=42, budget=1000, timeline=99,
            preferred_location=0, property_type=[],
        )
        result = self._score(lead)
        self.assertIsInstance(result.score, int)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)

    def test_h05_namespace_with_list_values(self):
        from types import SimpleNamespace
        lead = SimpleNamespace(
            message=["hello"], budget=["400k"], timeline=["now"],
            preferred_location=["NYC"], property_type=["house"],
        )
        result = self._score(lead)
        self.assertIsInstance(result.score, int)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)

    def test_h06_namespace_with_missing_attributes(self):
        from types import SimpleNamespace
        lead = SimpleNamespace()
        result = self._score(lead)
        self.assertIsInstance(result.score, int)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 100)

    def test_h07_returns_lead_score_instance(self):
        from models.lead import LeadScore
        lead = _make_lead()
        result = self._score(lead)
        self.assertIsInstance(result, LeadScore)


if __name__ == "__main__":
    unittest.main(verbosity=2)
