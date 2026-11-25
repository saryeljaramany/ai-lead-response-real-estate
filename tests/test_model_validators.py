"""
tests/test_model_validators.py

Unit and HTTP-level tests for non-string type handling and string normalisation
across all mode="before" field validators in models/lead.py and
models/communication.py.

Rules enforced:
- Pure ASCII, UTF-8 without BOM, LF only.
- main.py is never imported; HTTP check mounts only routers.ai.router on a
  throwaway FastAPI app.
- Standard library unittest with subTest.
"""

from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from models import EmailPayload, HubSpotContact, LeadInput, LeadScore, SMSPayload
import routers.ai as ai_router


def _valid_lead_input() -> dict:
    return {
        "name": "Sarah Mitchell",
        "email": "sarah.mitchell@example.com",
        "phone": "+14155550192",
        "message": "Looking for a three-bedroom home.",
    }


def _valid_lead_score() -> dict:
    return {
        "score": 85,
        "reasoning": "Strong buyer intent and clear timeline.",
        "suggested_action": "Call today.",
        "priority_level": "urgent",
    }


def _valid_email_payload() -> dict:
    return {
        "to_email": "sarah.mitchell@example.com",
        "to_name": "Sarah Mitchell",
        "subject": "Property Search Follow-Up",
        "body": "Hello Sarah, thank you for reaching out.",
        "lead_score": 85,
    }


def _valid_sms_payload() -> dict:
    return {
        "to_phone": "+14155550192",
        "message": "Hi Sarah, an agent will contact you shortly.",
        "lead_score": 85,
    }


def _valid_hubspot_contact() -> dict:
    return {
        "email": "sarah.mitchell@example.com",
        "firstname": "Sarah",
        "lastname": "Mitchell",
        "phone": "+14155550192",
        "lead_score": 85,
        "lead_status": "hot",
    }


class TestModelValidators(unittest.TestCase):
    """Tests for mode='before' validator type guards and string behaviour."""

    def test_non_string_inputs_raise_validation_error_never_attribute_error(self) -> None:
        """All 11 mode='before' validators reject non-strings with ValidationError."""
        model_fields = [
            (LeadInput, _valid_lead_input, "name"),
            (LeadInput, _valid_lead_input, "phone"),
            (LeadInput, _valid_lead_input, "message"),
            (LeadScore, _valid_lead_score, "priority_level"),
            (EmailPayload, _valid_email_payload, "subject"),
            (EmailPayload, _valid_email_payload, "body"),
            (SMSPayload, _valid_sms_payload, "to_phone"),
            (SMSPayload, _valid_sms_payload, "message"),
            (HubSpotContact, _valid_hubspot_contact, "firstname"),
            (HubSpotContact, _valid_hubspot_contact, "lastname"),
            (HubSpotContact, _valid_hubspot_contact, "lead_status"),
        ]
        bad_values = [123, 1.5, None, True, ["x"], {"a": 1}]

        for model_cls, factory, field_name in model_fields:
            for bad_value in bad_values:
                with self.subTest(
                    model=model_cls.__name__,
                    field=field_name,
                    value_type=type(bad_value).__name__,
                ):
                    kwargs = factory()
                    kwargs[field_name] = bad_value
                    try:
                        with self.assertRaises(ValidationError):
                            model_cls(**kwargs)
                    except AttributeError as exc:
                        self.fail(
                            f"{model_cls.__name__}.{field_name} raised AttributeError "
                            f"for {bad_value!r}: {exc}"
                        )

    def test_string_normalisation_and_rejection_regressions(self) -> None:
        """Valid strings normalise as before; invalid strings raise ValidationError."""
        with self.subTest("lead_input_name_strip"):
            kwargs = _valid_lead_input()
            kwargs["name"] = "  Sarah  "
            lead = LeadInput(**kwargs)
            self.assertEqual(lead.name, "Sarah")

        with self.subTest("lead_input_whitespace_name_rejected"):
            kwargs = _valid_lead_input()
            kwargs["name"] = "   "
            with self.assertRaises(ValidationError) as ctx:
                LeadInput(**kwargs)
            self.assertIn("must not be empty", str(ctx.exception))

        with self.subTest("hubspot_lead_status_normalised"):
            kwargs = _valid_hubspot_contact()
            kwargs["lead_status"] = "Hot"
            contact = HubSpotContact(**kwargs)
            self.assertEqual(contact.lead_status, "hot")

        with self.subTest("hubspot_lead_status_lukewarm_rejected"):
            kwargs = _valid_hubspot_contact()
            kwargs["lead_status"] = "lukewarm"
            with self.assertRaises(ValidationError):
                HubSpotContact(**kwargs)

        with self.subTest("lead_score_priority_level_normalised"):
            kwargs = _valid_lead_score()
            kwargs["priority_level"] = " Urgent "
            score = LeadScore(**kwargs)
            self.assertEqual(score.priority_level, "urgent")

        with self.subTest("sms_message_161_chars_rejected"):
            kwargs = _valid_sms_payload()
            kwargs["message"] = "a" * 161
            with self.assertRaises(ValidationError):
                SMSPayload(**kwargs)

        with self.subTest("sms_to_phone_short_rejected"):
            kwargs = _valid_sms_payload()
            kwargs["to_phone"] = "123"
            with self.assertRaises(ValidationError):
                SMSPayload(**kwargs)

    def test_http_ai_score_lead_with_non_string_name_returns_422_not_500(self) -> None:
        """POST /ai/score-lead with name=123 returns 422 instead of 500."""
        app = FastAPI()
        app.include_router(ai_router.router)
        client = TestClient(app, raise_server_exceptions=False)

        payload = _valid_lead_input()
        payload["name"] = 123
        resp = client.post("/ai/score-lead", json=payload)
        self.assertEqual(resp.status_code, 422)
