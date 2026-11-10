"""
tests/test_llm_service_links.py

Unit tests for LLM copy generation with booking-link, identifier,
and compliance text injection by code.
Verifies that the LLM produces prose only while code owns URLs,
recipient addressing, phone numbers, and compliance notices.

Pure ASCII source: all non-ASCII characters are written as \\uXXXX escapes.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from models import EmailPayload, LeadInput, LeadScore, SMSPayload
from services.calendly_service import (
    LINK_TOKEN,
    SMS_MAX_CHARS,
    SMS_OPT_OUT,
    _EMAIL_CTA_FALLBACK,
    plan_sms,
)
from services.llm_service import generate_email, generate_sms, score_lead

BASE = "https://calendly.com/real-estate-team/property-viewing"


class FakeClient:
    """Mock client that records create(**kwargs) and returns configured content."""

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
    """Standard sample lead for LLM service tests."""
    return LeadInput(
        name="Sarah Mitchell",
        email="sarah.mitchell@example.com",
        phone="+14155550192",
        message="Looking for a 3-bedroom home near good schools.",
    )


def _make_score() -> LeadScore:
    """Standard sample score for LLM service tests."""
    return LeadScore(
        score=87,
        category="hot",
        reasoning="Concrete timeline and clear budget.",
        suggested_action="Call the lead today.",
        priority_level="urgent",
    )


class TestLlmServiceLinks(unittest.TestCase):
    """Test suite for link, recipient, and compliance injection in llm_service."""

    def setUp(self):
        self.lead = _make_lead()
        self.score = _make_score()
        self.expected_email_url = (
            BASE
            + "?name=Sarah%20Mitchell&email=sarah.mitchell%40example.com"
            "&utm_source=lead_system&utm_medium=email&utm_campaign=ai_followup"
        )

    # -----------------------------------------------------------------------
    # 1. generate_email with CALENDLY_LINK=BASE
    # -----------------------------------------------------------------------

    def test_generate_email_with_calendly_link(self):
        """Model JSON has token + hallucinated URL; code injects link and clean addressing."""
        fake_reply = json.dumps({
            "subject": "Your Austin Home Search",
            "body": (
                "Hi Sarah,\n\n"
                "We found wonderful properties for you.\n"
                f"{LINK_TOKEN}\n"
                "Also see http://hallucinated-property.com/listings\n"
                "Best regards,\nAgent"
            ),
            "to_email": "wrong.hallucinated@example.com",
            "to_name": "Wrong Name",
            "lead_score": 12,
        })
        client = FakeClient(fake_reply)

        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                result = generate_email(self.lead, self.score)

        self.assertIsInstance(result, EmailPayload)
        self.assertEqual(result.calendly_link, self.expected_email_url)
        self.assertEqual(result.body.count(self.expected_email_url), 1)
        self.assertIn(f"\n{self.expected_email_url}\n", result.body)
        self.assertNotIn(LINK_TOKEN, result.body)
        self.assertNotIn("http://hallucinated-property.com", result.body)
        self.assertEqual(result.to_email, self.lead.email)
        self.assertEqual(result.to_name, self.lead.name)
        self.assertEqual(result.lead_score, self.score.score)

    # -----------------------------------------------------------------------
    # 2. Token missing from the body -> CTA fallback + link appended
    # -----------------------------------------------------------------------

    def test_generate_email_token_missing_appends_cta_fallback(self):
        """When the LLM omits LINK_TOKEN, fallback CTA and URL are appended."""
        fake_reply = json.dumps({
            "subject": "Your Home Search",
            "body": "Hi Sarah,\n\nLet us know when you would like to view homes.\nBest,\nTeam",
        })
        client = FakeClient(fake_reply)

        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                result = generate_email(self.lead, self.score)

        self.assertIn(_EMAIL_CTA_FALLBACK, result.body)
        self.assertIn(self.expected_email_url, result.body)
        self.assertEqual(result.calendly_link, self.expected_email_url)

    # -----------------------------------------------------------------------
    # 3. CALENDLY_LINK unset -> calendly_link None; no token and no URL in body
    # -----------------------------------------------------------------------

    def test_generate_email_without_link(self):
        """When CALENDLY_LINK is unset, tokens and hallucinated URLs are removed."""
        fake_reply = json.dumps({
            "subject": "Your Home Search",
            "body": (
                "Hi Sarah,\n\n"
                f"Please book here: {LINK_TOKEN}\n"
                "Or check https://evil.com/fake\n"
                "Best,\nTeam"
            ),
        })
        client = FakeClient(fake_reply)

        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                result = generate_email(self.lead, self.score)

        self.assertIsNone(result.calendly_link)
        self.assertNotIn(LINK_TOKEN, result.body)
        self.assertNotIn("https://evil.com/fake", result.body)

    # -----------------------------------------------------------------------
    # 4. Prompt content for email
    # -----------------------------------------------------------------------

    def test_generate_email_prompt_content(self):
        """Verify prompt rules for email: token instructed, no PII/URL leak."""
        client_with_link = FakeClient(
            json.dumps({"subject": "Sub", "body": f"Prose\n{LINK_TOKEN}"})
        )
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            with patch("services.llm_service._get_client", return_value=client_with_link):
                generate_email(self.lead, self.score)

        prompt_with_link = client_with_link.kwargs["messages"][1]["content"]
        self.assertIn(LINK_TOKEN, prompt_with_link)
        self.assertNotIn(BASE, prompt_with_link)
        self.assertNotIn(self.lead.email, prompt_with_link)
        self.assertNotIn(self.lead.phone, prompt_with_link)

        client_no_link = FakeClient(
            json.dumps({"subject": "Sub", "body": "Prose without token."})
        )
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client_no_link):
                generate_email(self.lead, self.score)

        prompt_no_link = client_no_link.kwargs["messages"][1]["content"]
        self.assertNotIn(LINK_TOKEN, prompt_no_link)
        self.assertIn("No booking link is available", prompt_no_link)
        self.assertIn("reply to this email", prompt_no_link)
        self.assertNotIn(self.lead.email, prompt_no_link)
        self.assertNotIn(self.lead.phone, prompt_no_link)

    # -----------------------------------------------------------------------
    # 5. Parsing edge cases and ValueError enforcement
    # -----------------------------------------------------------------------

    def test_generate_email_parsing_variants(self):
        """Fenced JSON is parsed; malformed or missing fields raise ValueError."""
        # Fenced markdown is stripped cleanly
        fenced_reply = (
            "```json\n"
            + json.dumps({"subject": "Homes", "body": f"Body\n{LINK_TOKEN}"})
            + "\n```"
        )
        client = FakeClient(fenced_reply)
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                res = generate_email(self.lead, self.score)
                self.assertEqual(res.subject, "Homes")

        invalid_replies = [
            "not a valid json string at all",
            json.dumps([1, 2, 3]),
            json.dumps({"body": "Body without subject"}),
            json.dumps({"subject": "   ", "body": "Body"}),
            json.dumps({"subject": "Subject without body"}),
            json.dumps({"subject": "Subject", "body": "   "}),
            json.dumps({"subject": 123, "body": "Body"}),
            json.dumps({"subject": "Subject", "body": 456}),
        ]
        for bad_reply in invalid_replies:
            with self.subTest(bad_reply=bad_reply):
                bad_client = FakeClient(bad_reply)
                with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
                    with patch("services.llm_service._get_client", return_value=bad_client):
                        with self.assertRaises(ValueError):
                            generate_email(self.lead, self.score)

    # -----------------------------------------------------------------------
    # 6. generate_sms with a link
    # -----------------------------------------------------------------------

    def test_generate_sms_with_link(self):
        """Model message contains URL + opt-out; code sanitises and appends plan."""
        fake_reply = json.dumps({
            "message": (
                "Hi Sarah, visit https://scam.com Reply STOP to opt out. "
                "Book a viewing here:"
            ),
            "to_phone": "+19999999999",
            "lead_score": 1,
        })
        client = FakeClient(fake_reply)

        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            plan = plan_sms(self.lead)
            with patch("services.llm_service._get_client", return_value=client):
                result = generate_sms(self.lead, self.score)

        self.assertIsInstance(result, SMSPayload)
        self.assertLessEqual(len(result.message), SMS_MAX_CHARS)
        expected_suffix = " " + plan.link + " " + SMS_OPT_OUT
        self.assertTrue(result.message.endswith(expected_suffix))
        self.assertEqual(result.message.count(SMS_OPT_OUT), 1)
        self.assertIn(plan.link, result.message)
        self.assertNotIn("https://scam.com", result.message)
        self.assertEqual(result.to_phone, self.lead.phone)
        self.assertEqual(result.lead_score, self.score.score)

        prompt = client.kwargs["messages"][1]["content"]
        self.assertIn(str(plan.text_budget), prompt)
        self.assertNotIn(self.lead.phone, prompt)
        self.assertIn(
            "Do NOT include any URL and do NOT include an opt-out or STOP instruction",
            prompt,
        )
        self.assertIn("Book a viewing here", prompt)

        # Assert there is no "text" fallback: {"text": "hi"} raises ValueError
        text_only_client = FakeClient(json.dumps({"text": "hi"}))
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            with patch("services.llm_service._get_client", return_value=text_only_client):
                with self.assertRaises(ValueError):
                    generate_sms(self.lead, self.score)

    # -----------------------------------------------------------------------
    # 7. generate_sms without a link
    # -----------------------------------------------------------------------

    def test_generate_sms_without_link(self):
        """When CALENDLY_LINK is unset, SMS ends with opt-out only, no URL."""
        fake_reply = json.dumps({
            "message": "Hi Sarah, an agent will contact you shortly to assist.",
        })
        client = FakeClient(fake_reply)

        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service._get_client", return_value=client):
                result = generate_sms(self.lead, self.score)

        self.assertLessEqual(len(result.message), SMS_MAX_CHARS)
        self.assertTrue(result.message.endswith(" " + SMS_OPT_OUT))
        self.assertNotIn("http", result.message)
        self.assertNotIn("calendly.com", result.message)

        prompt = client.kwargs["messages"][1]["content"]
        self.assertIn("Do not mention booking links or scheduling", prompt)

    # -----------------------------------------------------------------------
    # 8. Over-long model text is cut at a word boundary
    # -----------------------------------------------------------------------

    def test_generate_sms_overlong_cut_at_word_boundary(self):
        """Over-long model output is truncated cleanly at word boundary."""
        overlong_body = (
            "This is an extraordinarily long text message that far exceeds "
            "the planned budget and keeps rambling on about houses, apartments, "
            "neighborhoods, mortgages, and various other real estate topics "
            "without ever pausing."
        )
        fake_reply = json.dumps({"message": overlong_body})
        client = FakeClient(fake_reply)

        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            plan = plan_sms(self.lead)
            with patch("services.llm_service._get_client", return_value=client):
                result = generate_sms(self.lead, self.score)

        self.assertLessEqual(len(result.message), SMS_MAX_CHARS)
        self.assertTrue(result.message.endswith(plan.suffix))
        self.assertEqual(result.message.count(SMS_OPT_OUT), 1)
        self.assertEqual(result.to_phone, self.lead.phone)
        self.assertEqual(result.lead_score, self.score.score)

        # Confirm truncation occurred at a word boundary (no trailing partial word)
        prose_part = result.message[: -len(plan.suffix)]
        self.assertFalse(prose_part.endswith(" "))
        self.assertFalse(prose_part.endswith(","))

    # -----------------------------------------------------------------------
    # 9. Regression: score_lead still parses fake JSON reply
    # -----------------------------------------------------------------------

    def test_score_lead_regression(self):
        """Verify score_lead remains functional after all Phase 8 modifications."""
        score_json = json.dumps({
            "score": 87,
            "category": "hot",
            "reasoning": "High budget and concrete timeline.",
            "suggested_action": "Contact lead today and schedule viewing.",
            "priority_level": "urgent",
        })
        client = FakeClient(score_json)

        with patch("services.llm_service._get_client", return_value=client):
            result = score_lead(self.lead)

        self.assertIsInstance(result, LeadScore)
        self.assertEqual(result.score, 87)
        self.assertEqual(result.category, "hot")
        self.assertEqual(result.priority_level, "urgent")


if __name__ == "__main__":
    unittest.main()
