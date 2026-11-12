"""
tests/test_llm_scoring_privacy.py

Privacy and regression unit tests for score_lead, generate_email, and generate_sms
in services/llm_service.py. Verifies that contact details (email and phone number)
never reach the AI model in scoring prompts, and verifies that scoring inputs and
valid LeadScore outputs are preserved.

Pure ASCII source: all text is strictly ASCII.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import LeadInput, LeadScore
from services.llm_service import generate_email, generate_sms, score_lead


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


def _make_full_lead() -> LeadInput:
    """Sample full lead with all optional fields provided."""
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


def _make_minimal_lead() -> LeadInput:
    """Sample lead with only required contact and message fields."""
    return LeadInput(
        name="Sarah Mitchell",
        email="sarah.mitchell@example.com",
        phone="+14155550192",
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


class TestLlmScoringPrivacy(unittest.TestCase):
    """Test suite verifying lead contact privacy and prompt safety across LLM services."""

    def setUp(self):
        self.lead = _make_full_lead()
        self.minimal_lead = _make_minimal_lead()
        self.score = _make_score()

    # -----------------------------------------------------------------------
    # A. Contact details never reach the model
    # -----------------------------------------------------------------------

    def test_contact_details_never_reach_model(self):
        """Joining all recorded messages after score_lead contains no email or phone details."""
        client = FakeClient(SCORE_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                score_lead(self.lead)

        joined_content = " ".join(
            msg["content"] for msg in client.kwargs.get("messages", [])
        )

        forbidden_items = [
            "sarah.mitchell@example.com",
            "+14155550192",
            "4155550192",
            "Email:",
            "Phone:",
        ]
        for item in forbidden_items:
            with self.subTest(forbidden=item):
                self.assertNotIn(item, joined_content)

    # -----------------------------------------------------------------------
    # B. Scoring inputs are still sent
    # -----------------------------------------------------------------------

    def test_scoring_inputs_are_still_sent(self):
        """User prompt retains name, property type, location, budget, timeline, and message."""
        client = FakeClient(SCORE_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                score_lead(self.lead)

        user_content = next(
            msg["content"]
            for msg in client.kwargs.get("messages", [])
            if msg.get("role") == "user"
        )

        expected_inputs = [
            "Sarah Mitchell",
            "house",
            "Austin, TX",
            "$450k-$550k",
            "within 3 months",
            "Looking for a three-bedroom house near good schools.",
        ]
        for item in expected_inputs:
            with self.subTest(expected=item):
                self.assertIn(item, user_content)

    # -----------------------------------------------------------------------
    # C. Optional fields missing
    # -----------------------------------------------------------------------

    def test_optional_fields_missing(self):
        """A minimal lead displays 'Not provided' for missing fields while omitting contact info."""
        client = FakeClient(SCORE_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                score_lead(self.minimal_lead)

        user_content = next(
            msg["content"]
            for msg in client.kwargs.get("messages", [])
            if msg.get("role") == "user"
        )
        self.assertIn("Not provided", user_content)

        joined_content = " ".join(
            msg["content"] for msg in client.kwargs.get("messages", [])
        )
        forbidden_items = [
            "sarah.mitchell@example.com",
            "+14155550192",
            "4155550192",
            "Email:",
            "Phone:",
        ]
        for item in forbidden_items:
            with self.subTest(forbidden=item):
                self.assertNotIn(item, joined_content)

    # -----------------------------------------------------------------------
    # D. Result still valid
    # -----------------------------------------------------------------------

    def test_result_still_valid(self):
        """score_lead correctly parses model response into a LeadScore with expected values."""
        client = FakeClient(SCORE_REPLY)
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                result = score_lead(self.lead)

        self.assertIsInstance(result, LeadScore)
        self.assertEqual(result.score, 87)
        self.assertEqual(result.category, "hot")

    # -----------------------------------------------------------------------
    # E. Regression for generate_email and generate_sms
    # -----------------------------------------------------------------------

    def test_regression_email_and_sms_contact_privacy(self):
        """generate_email and generate_sms never send email or phone in messages to the model."""
        email_reply = json.dumps({
            "subject": "Hello",
            "body": "Hi Sarah, let us talk.",
        })
        sms_reply = json.dumps({
            "message": "Hi Sarah, an agent will contact you shortly.",
        })

        test_channels = [
            (
                "email",
                email_reply,
                lambda: generate_email(self.lead, self.score),
            ),
            (
                "sms",
                sms_reply,
                lambda: generate_sms(self.lead, self.score),
            ),
        ]

        forbidden_items = [
            "sarah.mitchell@example.com",
            "+14155550192",
            "4155550192",
        ]

        with patch.dict(os.environ, {}, clear=True):
            for channel, reply, call in test_channels:
                with self.subTest(channel=channel):
                    client = FakeClient(reply)
                    with patch(
                        "services.llm_service._get_client", return_value=client
                    ):
                        call()

                    joined_content = " ".join(
                        msg["content"]
                        for msg in client.kwargs.get("messages", [])
                    )
                    for item in forbidden_items:
                        with self.subTest(channel=channel, forbidden=item):
                            self.assertNotIn(item, joined_content)


if __name__ == "__main__":
    unittest.main()
