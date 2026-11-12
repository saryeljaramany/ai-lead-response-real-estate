"""
tests/test_llm_model_selection.py

Unit tests for dynamic OpenRouter model selection via OPENROUTER_MODEL.
Verifies call-time resolution, fallback to _DEFAULT_MODEL, and wiring
into score_lead, generate_email, and generate_sms.

Pure ASCII source: all non-ASCII characters are written as \\uXXXX escapes.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import LeadInput, LeadScore
from services.llm_service import (
    _DEFAULT_MODEL,
    _get_model,
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
    """Standard sample lead for model selection tests."""
    return LeadInput(
        name="Sarah Mitchell",
        email="sarah.mitchell@example.com",
        phone="+14155550192",
        message="Looking for a three-bedroom house in Austin.",
    )


def _make_score() -> LeadScore:
    """Standard sample score for model selection tests."""
    return LeadScore(
        score=87,
        reasoning="r",
        suggested_action="a",
        priority_level="urgent",
    )


class TestLlmModelSelection(unittest.TestCase):
    """Test suite for OPENROUTER_MODEL resolution and wiring in services/llm_service.py."""

    def setUp(self):
        self.lead = _make_lead()
        self.score = _make_score()

    # -----------------------------------------------------------------------
    # A. Import safety
    # -----------------------------------------------------------------------

    def test_import_with_empty_environment(self):
        """Importing services.llm_service with an empty environment does not raise."""
        with patch.dict(os.environ, {}, clear=True):
            import services.llm_service as svc

            self.assertTrue(hasattr(svc, "_get_model"))
            self.assertTrue(hasattr(svc, "_DEFAULT_MODEL"))

    # -----------------------------------------------------------------------
    # B. _get_model table
    # -----------------------------------------------------------------------

    def test_default_model_constant_value(self):
        """_DEFAULT_MODEL itself must equal 'openrouter/free'."""
        self.assertEqual(_DEFAULT_MODEL, "openrouter/free")

    def test_get_model_resolution_table(self):
        """_get_model resolves valid stripped values or defaults on blank/unset."""
        cases = [
            (None, "openrouter/free"),
            ("", "openrouter/free"),
            ("   ", "openrouter/free"),
            ("\t\n", "openrouter/free"),
            (
                "nvidia/nemotron-3.5-lightning:free",
                "nvidia/nemotron-3.5-lightning:free",
            ),
            ("  vendor/model:free  ", "vendor/model:free"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw, expected=expected):
                env = {} if raw is None else {"OPENROUTER_MODEL": raw}
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(_get_model(), expected)

    # -----------------------------------------------------------------------
    # C. Call-time read
    # -----------------------------------------------------------------------

    def test_call_time_read_dynamic_update(self):
        """_get_model reads at call time and reflects environment changes immediately."""
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "a/one:free"}, clear=True):
            self.assertEqual(_get_model(), "a/one:free")
            os.environ["OPENROUTER_MODEL"] = "b/two:free"
            self.assertEqual(_get_model(), "b/two:free")

    # -----------------------------------------------------------------------
    # D. Wiring of all three functions
    # -----------------------------------------------------------------------

    def test_wiring_all_three_functions(self):
        """score_lead, generate_email, and generate_sms pass the resolved model to create()."""
        score_reply = json.dumps({
            "score": 87,
            "category": "hot",
            "reasoning": "r",
            "suggested_action": "a",
            "priority_level": "urgent",
        })
        email_reply = json.dumps({
            "subject": "Hello",
            "body": "Hi Sarah, let us talk.",
        })
        sms_reply = json.dumps({
            "message": "Hi Sarah, an agent will contact you shortly.",
        })

        test_cases = [
            ("score_lead", score_reply, lambda: score_lead(self.lead)),
            (
                "generate_email",
                email_reply,
                lambda: generate_email(self.lead, self.score),
            ),
            (
                "generate_sms",
                sms_reply,
                lambda: generate_sms(self.lead, self.score),
            ),
        ]

        # (1) OPENROUTER_MODEL="vendor/model:free" -> recorded kwargs["model"] == "vendor/model:free"
        with patch.dict(
            os.environ, {"OPENROUTER_MODEL": "vendor/model:free"}, clear=True
        ):
            for name, reply, call in test_cases:
                with self.subTest(fn=name, env="configured"):
                    client = FakeClient(reply)
                    with patch(
                        "services.llm_service._get_client", return_value=client
                    ):
                        call()
                        self.assertEqual(
                            client.kwargs["model"], "vendor/model:free"
                        )

        # (2) Variable unset -> recorded kwargs["model"] == "openrouter/free"
        with patch.dict(os.environ, {}, clear=True):
            for name, reply, call in test_cases:
                with self.subTest(fn=name, env="unset"):
                    client = FakeClient(reply)
                    with patch(
                        "services.llm_service._get_client", return_value=client
                    ):
                        call()
                        self.assertEqual(
                            client.kwargs["model"], "openrouter/free"
                        )

    # -----------------------------------------------------------------------
    # E. Model identifier never leaked in prompt
    # -----------------------------------------------------------------------

    def test_model_identifier_not_leaked_in_prompt(self):
        """With OPENROUTER_MODEL set, the model identifier never appears in prompt content."""
        model_name = "vendor/model:free"
        score_reply = json.dumps({
            "score": 87,
            "category": "hot",
            "reasoning": "r",
            "suggested_action": "a",
            "priority_level": "urgent",
        })
        email_reply = json.dumps({
            "subject": "Hello",
            "body": "Hi Sarah, let us talk.",
        })
        sms_reply = json.dumps({
            "message": "Hi Sarah, an agent will contact you shortly.",
        })

        test_cases = [
            ("score_lead", score_reply, lambda: score_lead(self.lead)),
            (
                "generate_email",
                email_reply,
                lambda: generate_email(self.lead, self.score),
            ),
            (
                "generate_sms",
                sms_reply,
                lambda: generate_sms(self.lead, self.score),
            ),
        ]

        with patch.dict(
            os.environ, {"OPENROUTER_MODEL": model_name}, clear=True
        ):
            for name, reply, call in test_cases:
                with self.subTest(fn=name):
                    client = FakeClient(reply)
                    with patch(
                        "services.llm_service._get_client", return_value=client
                    ):
                        call()
                        for msg in client.kwargs["messages"]:
                            self.assertNotIn(model_name, msg["content"])


if __name__ == "__main__":
    unittest.main()
