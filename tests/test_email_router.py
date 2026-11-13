"""
tests/test_email_router.py

Integration tests for the email HTTP endpoints (routers/email.py).

Rules enforced:
- Only unittest, unittest.mock, and fastapi.testclient.TestClient.
- No network access, no .env file read.
- main.py is never imported.
- Router tests build a throwaway FastAPI app: app = FastAPI(); app.include_router(email_router.router).
- Table-driven tests use subTest.
- Pure ASCII and saved as UTF-8 without BOM.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.email as email_router
from services.gmail_service import EmailSendError


def _build_test_client() -> TestClient:
    """Build a throwaway FastAPI test client with only the email router included."""
    app = FastAPI()
    app.include_router(email_router.router)
    return TestClient(app, raise_server_exceptions=False)


class TestEmailRouter(unittest.TestCase):
    """Automated test suite for routers/email.py."""

    def setUp(self) -> None:
        self.client = _build_test_client()
        self.valid_payload = {
            "to_email": "sarah.mitchell@example.com",
            "to_name": "Sarah Mitchell",
            "subject": "Your Austin Home Search",
            "body": "Hi Sarah, thank you for reaching out.",
            "lead_score": 87,
        }

    # -----------------------------------------------------------------------
    # 1. Patched send_email returns dict -> 200 with same body
    # -----------------------------------------------------------------------

    def test_send_email_success_returns_200(self) -> None:
        """POST /email/send returns 200 with the exact dictionary returned by send_email."""
        expected_body = {
            "message": "Email sent successfully",
            "to": self.valid_payload["to_email"],
            "subject": self.valid_payload["subject"],
            "message_id": "msg_xyz123",
            "dry_run": False,
        }
        with patch("routers.email.send_email") as mock_send:
            mock_send.return_value = expected_body
            response = self.client.post("/email/send", json=self.valid_payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), expected_body)

    # -----------------------------------------------------------------------
    # 2. Error mapping: recipient -> 422, transient -> 502, config -> 500, etc.
    # -----------------------------------------------------------------------

    def test_send_email_error_mapping(self) -> None:
        """EmailSendError maps to 422/502/500, RuntimeError to 500, and generic Exception masks details."""
        error_cases = [
            ("recipient -> 422", EmailSendError("Bad recipient", category="recipient"), 422, "Bad recipient"),
            ("transient -> 502", EmailSendError("Rate limit", category="transient"), 502, "Rate limit"),
            ("config -> 500", EmailSendError("Invalid scope", category="config"), 500, "Invalid scope"),
            ("RuntimeError -> 500", RuntimeError("token.json missing"), 500, "token.json missing"),
        ]

        for label, exc, expected_status, expected_detail in error_cases:
            with self.subTest(case=label):
                with patch("routers.email.send_email") as mock_send:
                    mock_send.side_effect = exc
                    response = self.client.post("/email/send", json=self.valid_payload)
                    self.assertEqual(response.status_code, expected_status)
                    self.assertEqual(response.json().get("detail"), expected_detail)

    def test_send_email_generic_exception_masks_detail(self) -> None:
        """Unexpected Exception('boom') returns 500 with a generic detail string not containing 'boom'."""
        with patch("routers.email.send_email") as mock_send:
            mock_send.side_effect = Exception("boom")
            response = self.client.post("/email/send", json=self.valid_payload)
            self.assertEqual(response.status_code, 500)
            detail = response.json().get("detail", "")
            self.assertNotIn("boom", detail)
            self.assertIn("Unexpected error while sending the email", detail)

    # -----------------------------------------------------------------------
    # 3. Invalid payload -> 422
    # -----------------------------------------------------------------------

    def test_invalid_payload_empty_subject_returns_422(self) -> None:
        """POST /email/send with empty subject fails model validation and returns 422."""
        invalid_payload = dict(self.valid_payload, subject="")
        response = self.client.post("/email/send", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    # -----------------------------------------------------------------------
    # 4. Handler synchronicity check (not coroutine functions)
    # -----------------------------------------------------------------------

    def test_endpoint_handlers_are_sync(self) -> None:
        """Endpoint handlers must be plain def (not async def) to prevent blocking the event loop."""
        handlers = [
            ("send_email_endpoint", email_router.send_email_endpoint),
            ("auth", email_router.auth),
            ("auth_callback", email_router.auth_callback),
        ]
        for name, func in handlers:
            with self.subTest(handler=name):
                self.assertFalse(
                    inspect.iscoroutinefunction(func),
                    f"{name} should be a plain synchronous function",
                )

    # -----------------------------------------------------------------------
    # 5. Callback regression tests
    # -----------------------------------------------------------------------

    def test_callback_denied_returns_400_without_calling_exchange(self) -> None:
        """GET /email/auth/callback?error=access_denied returns 400 without calling exchange_code."""
        with patch("routers.email.exchange_code") as mock_exchange:
            response = self.client.get("/email/auth/callback?error=access_denied")
            self.assertEqual(response.status_code, 400)
            self.assertIn("Authorization denied", response.json().get("detail", ""))
            mock_exchange.assert_not_called()

    def test_callback_missing_code_returns_400_without_calling_exchange(self) -> None:
        """GET /email/auth/callback without query params returns 400 without calling exchange_code."""
        with patch("routers.email.exchange_code") as mock_exchange:
            response = self.client.get("/email/auth/callback")
            self.assertEqual(response.status_code, 400)
            self.assertIn("Missing 'code'", response.json().get("detail", ""))
            mock_exchange.assert_not_called()


if __name__ == "__main__":
    unittest.main()
