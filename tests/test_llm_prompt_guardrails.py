"""
tests/test_llm_prompt_guardrails.py

Unit tests for the honesty rule guardrail in services/llm_service.py.
Verifies that _HONESTY_RULE is well-formed, injected into email and SMS
user prompts on dedicated guideline lines, omitted from score_lead prompts,
and absent from system messages across all functions.

Pure ASCII source: all text is strictly ASCII.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import LeadInput, LeadScore
from services.llm_service import (
    _HONESTY_RULE,
    generate_email,
    generate_sms,
    score_lead,
)


class FakeClient:
    """Mock client that records chat.completions.create kwargs and returns configured content."""

    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=self._outer.content)
                        )
                    ]
                )

        class _Chat:
            def __init__(self, outer):
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


def _make_lead() -> LeadInput:
    """Sample lead with all fields provided."""
    return LeadInput(
        name="Sarah Mitchell",
        email="sarah.mitchell@example.com",
        phone="+14155550192",
        property_type="house",
        preferred_location="Austin, TX",
        budget="$450k-$550k",
        timeline="within 3 months",
        message="Looking for a three-bedroom house near good schools.",
    )


def _make_score() -> LeadScore:
    """Sample LeadScore for follow-up message generation."""
    return LeadScore(
        score=87,
        reasoning="r",
        suggested_action="a",
        priority_level="urgent",
    )


SCORE_REPLY = json.dumps({
    "score": 87,
    "category": "hot",
    "reasoning": "r",
    "suggested_action": "a",
    "priority_level": "urgent",
})

EMAIL_REPLY = json.dumps({
    "subject": "Hello",
    "body": "Hi Sarah, let us talk.",
})

SMS_REPLY = json.dumps({
    "message": "Hi Sarah, an agent will contact you shortly.",
})


class TestLlmPromptGuardrails(unittest.TestCase):
    """Test suite verifying _HONESTY_RULE definition, placement, and absence where prohibited."""

    def setUp(self):
        self.lead = _make_lead()
        self.score = _make_score()

    # -----------------------------------------------------------------------
    # A. Rule text content and clauses
    # -----------------------------------------------------------------------

    def test_honesty_rule_text(self):
        """_HONESTY_RULE is a non-empty string and contains all required clauses."""
        self.assertIsInstance(_HONESTY_RULE, str)
        self.assertTrue(len(_HONESTY_RULE.strip()) > 0)

        clauses = [
            "already found",
            "curated",
            "shortlisted",
            "prepared",
            "following up",
            "arranging a viewing",
            "never invent prices",
            "never call the person a",
            "never mention their score",
        ]
        rule_lower = _HONESTY_RULE.lower()
        for clause in clauses:
            with self.subTest(clause=clause):
                self.assertIn(clause, rule_lower)

    # -----------------------------------------------------------------------
    # B. Email prompt contains honesty rule
    # -----------------------------------------------------------------------

    def test_email_prompt_contains_honesty_rule(self):
        """generate_email includes _HONESTY_RULE exactly once in user prompt on a bullet line."""
        client = FakeClient(EMAIL_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                generate_email(self.lead, self.score)

        user_content = next(
            msg["content"]
            for msg in client.kwargs.get("messages", [])
            if msg.get("role") == "user"
        )

        self.assertEqual(user_content.count(_HONESTY_RULE), 1)

        matching_lines = [
            line
            for line in user_content.splitlines()
            if line.startswith("- ") and _HONESTY_RULE in line
        ]
        self.assertEqual(len(matching_lines), 1)

    def test_email_prompt_excludes_ready_to_act(self):
        """With environment cleared and score 87, email prompt excludes 'ready-to-act'."""
        client = FakeClient(EMAIL_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                generate_email(self.lead, self.score)

        user_content = next(
            msg["content"]
            for msg in client.kwargs.get("messages", [])
            if msg.get("role") == "user"
        )

        self.assertNotIn("ready-to-act", user_content.lower())
        self.assertEqual(user_content.count(_HONESTY_RULE), 1)

    # -----------------------------------------------------------------------
    # C. SMS prompt contains honesty rule and retains existing guardrails
    # -----------------------------------------------------------------------

    def test_sms_prompt_contains_honesty_rule_and_retains_guardrails(self):
        """generate_sms includes _HONESTY_RULE on a bullet line and keeps budget/URL rules."""
        client = FakeClient(SMS_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                generate_sms(self.lead, self.score)

        user_content = next(
            msg["content"]
            for msg in client.kwargs.get("messages", [])
            if msg.get("role") == "user"
        )

        self.assertEqual(user_content.count(_HONESTY_RULE), 1)

        matching_lines = [
            line
            for line in user_content.splitlines()
            if line.startswith("- ") and _HONESTY_RULE in line
        ]
        self.assertEqual(len(matching_lines), 1)
        self.assertIn("CRITICAL", user_content)
        self.assertIn("Do NOT include any URL", user_content)

    # -----------------------------------------------------------------------
    # D. Scoring prompt excludes honesty rule
    # -----------------------------------------------------------------------

    def test_scoring_prompt_excludes_honesty_rule(self):
        """score_lead never includes _HONESTY_RULE or its signature phrases."""
        client = FakeClient(SCORE_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                score_lead(self.lead)

        all_content = " ".join(
            msg["content"] for msg in client.kwargs.get("messages", [])
        )
        self.assertNotIn(_HONESTY_RULE, all_content)
        self.assertNotIn("already found", all_content.lower())
        self.assertNotIn("curated", all_content.lower())

    # -----------------------------------------------------------------------
    # E. Rule is not part of system message in any of the three calls
    # -----------------------------------------------------------------------

    def test_system_message_excludes_honesty_rule(self):
        """System message in score_lead, generate_email, and generate_sms never contains the rule."""
        call_scenarios = [
            ("score_lead", SCORE_REPLY, lambda: score_lead(self.lead)),
            (
                "generate_email",
                EMAIL_REPLY,
                lambda: generate_email(self.lead, self.score),
            ),
            (
                "generate_sms",
                SMS_REPLY,
                lambda: generate_sms(self.lead, self.score),
            ),
        ]

        with patch.dict(os.environ, {}, clear=True):
            for name, reply, call_fn in call_scenarios:
                with self.subTest(scenario=name):
                    client = FakeClient(reply)
                    with patch(
                        "services.llm_service._get_client", return_value=client
                    ):
                        call_fn()

                    system_content = next(
                        msg["content"]
                        for msg in client.kwargs.get("messages", [])
                        if msg.get("role") == "system"
                    )
                    self.assertNotIn(_HONESTY_RULE, system_content)


if __name__ == "__main__":
    unittest.main()
