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
from services.security import SecurityConfigError


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

    # -----------------------------------------------------------------------
    # 6. OAuth state generation and validation in email router
    # -----------------------------------------------------------------------

    def test_auth_generates_state_and_calls_get_auth_url(self) -> None:
        """GET /email/auth creates OAuth state and passes it to get_auth_url."""
        mock_state = "mocked_oauth_state_payload_123"
        expected_url = "https://accounts.google.com/o/oauth2/auth?state=" + mock_state

        with patch("routers.email.make_oauth_state", return_value=mock_state) as mock_make_state, \
             patch("routers.email.get_auth_url", return_value=expected_url) as mock_get_url:
            response = self.client.get("/email/auth")
            self.assertEqual(response.status_code, 200)
            mock_make_state.assert_called_once()
            mock_get_url.assert_called_once_with(mock_state)
            self.assertEqual(response.json().get("auth_url"), expected_url)

    def test_auth_security_config_error_returns_500(self) -> None:
        """GET /email/auth maps SecurityConfigError to HTTP 500 with its message."""
        err_msg = "API_SECRET_KEY is missing or shorter than 24 characters."
        with patch("routers.email.make_oauth_state", side_effect=SecurityConfigError(err_msg)):
            response = self.client.get("/email/auth")
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json().get("detail"), err_msg)

    def test_callback_missing_or_invalid_state_returns_400_without_calling_exchange(self) -> None:
        """GET /email/auth/callback with code but missing/invalid state returns 400 without exchanging."""
        with patch("routers.email.exchange_code") as mock_exchange:
            # Code present, missing state
            with patch("routers.email.verify_oauth_state", return_value=False) as mock_verify:
                resp_no_state = self.client.get("/email/auth/callback?code=valid_code_123")
                self.assertEqual(resp_no_state.status_code, 400)
                self.assertIn("Invalid or expired OAuth state", resp_no_state.json().get("detail", ""))
                mock_verify.assert_called_once_with(None)
                mock_exchange.assert_not_called()

            # Code present, invalid state
            mock_exchange.reset_mock()
            with patch("routers.email.verify_oauth_state", return_value=False) as mock_verify:
                resp_bad_state = self.client.get(
                    "/email/auth/callback?code=valid_code_123&state=bad.state.token"
                )
                self.assertEqual(resp_bad_state.status_code, 400)
                self.assertIn("Invalid or expired OAuth state", resp_bad_state.json().get("detail", ""))
                mock_verify.assert_called_once_with("bad.state.token")
                mock_exchange.assert_not_called()

    def test_callback_valid_state_exchanges_code_returns_200(self) -> None:
        """GET /email/auth/callback with code and valid state exchanges code and returns 200."""
        valid_state = "valid.oauth.state.token"
        with patch("routers.email.verify_oauth_state", return_value=True) as mock_verify, \
             patch("routers.email.exchange_code") as mock_exchange:
            mock_exchange.return_value = None
            response = self.client.get(
                f"/email/auth/callback?code=google_auth_code_xyz&state={valid_state}"
            )
            self.assertEqual(response.status_code, 200)
            mock_verify.assert_called_once_with(valid_state)
            mock_exchange.assert_called_once_with("google_auth_code_xyz")
            self.assertIn("Authorization successful", response.json().get("message", ""))


if __name__ == "__main__":
    unittest.main()
