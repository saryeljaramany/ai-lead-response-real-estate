"""
tests/test_gmail_service.py

Unit tests for services/gmail_service.py.

Rules enforced:
- Only unittest and unittest.mock.
- No network access, no .env file read.
- main.py is never imported.
- All environment-dependent tests use patch.dict(os.environ, ..., clear=True).
- Table-driven tests use subTest.
- Constructors (HttpError, httplib2.Response) are verified via inspect.signature.
- Pure ASCII and saved as UTF-8 without BOM.
"""

from __future__ import annotations

import base64
import email
import importlib
import inspect
import json
import os
from pathlib import Path
import ssl
import unittest
from unittest.mock import MagicMock, patch

import google.auth.exceptions
from google.auth.exceptions import RefreshError, TransportError
from googleapiclient.errors import HttpError
import httplib2

# ---------------------------------------------------------------------------
# Confirm third-party constructor signatures before tests run
# ---------------------------------------------------------------------------

_HTTP_ERROR_SIG = inspect.signature(HttpError.__init__)
assert "resp" in _HTTP_ERROR_SIG.parameters, "HttpError.__init__ missing 'resp' param"
assert "content" in _HTTP_ERROR_SIG.parameters, "HttpError.__init__ missing 'content' param"

_RESPONSE_SIG = inspect.signature(httplib2.Response.__init__)
assert "info" in _RESPONSE_SIG.parameters, "httplib2.Response.__init__ missing 'info' param"


def _make_http_error(status: int, content: bytes = b"{}") -> HttpError:
    """Construct an HttpError using confirmed constructor signature."""
    resp = httplib2.Response({"status": status})
    return HttpError(resp, content)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestGmailService(unittest.TestCase):
    """Automated test suite for services/gmail_service.py."""

    # -----------------------------------------------------------------------
    # A. Import safety & ASCII encoding
    # -----------------------------------------------------------------------

    def test_import_safety(self) -> None:
        """Reloading the module under an empty environment must not raise."""
        with patch.dict(os.environ, {}, clear=True):
            import services.gmail_service as gs
            importlib.reload(gs)

    def test_source_is_ascii(self) -> None:
        """Source code of services/gmail_service.py must decode as pure ASCII."""
        source_path = (
            Path(__file__).resolve().parent.parent / "services" / "gmail_service.py"
        )
        content = source_path.read_bytes()
        try:
            content.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"services/gmail_service.py is not pure ASCII: {exc}")

    # -----------------------------------------------------------------------
    # B. _is_dry_run table test
    # -----------------------------------------------------------------------

    def test_is_dry_run_truthy_values(self) -> None:
        """Unset, empty, or non-falsy values must evaluate to True (safe dry-run)."""
        import services.gmail_service as gs

        truthy_cases = [
            ("unset", None),
            ("empty", ""),
            ("whitespace", "   "),
            ("true", "true"),
            ("TRUE", "TRUE"),
            ("yes", "yes"),
            ("garbage", "garbage"),
            ("1", "1"),
        ]
        for name, value in truthy_cases:
            with self.subTest(case=name):
                env = {} if value is None else {"GMAIL_DRY_RUN": value}
                with patch.dict(os.environ, env, clear=True):
                    self.assertTrue(gs._is_dry_run())

    def test_is_dry_run_falsy_values(self) -> None:
        """Explicit falsy values ('false', ' FALSE ', '0', 'no', 'off') evaluate to False."""
        import services.gmail_service as gs

        falsy_cases = [
            ("false", "false"),
            (" FALSE ", " FALSE "),
            ("0", "0"),
            ("no", "no"),
            ("off", "off"),
        ]
        for name, value in falsy_cases:
            with self.subTest(case=name):
                with patch.dict(os.environ, {"GMAIL_DRY_RUN": value}, clear=True):
                    self.assertFalse(gs._is_dry_run())

    # -----------------------------------------------------------------------
    # C. Dry-run execution
    # -----------------------------------------------------------------------

    def test_dry_run_send_email(self) -> None:
        """Dry-run returns immediately without credentials, token.json, or network."""
        import services.gmail_service as gs
        from models import EmailPayload

        payload = EmailPayload(
            to_email="lead@example.com",
            to_name="Lead Name",
            subject="First Line\nSecond   Line",
            body="Hello, this is a dry run body.",
            lead_score=75,
        )

        with patch.dict(os.environ, {}, clear=True):
            with patch("services.gmail_service._load_credentials") as mock_creds, \
                 patch("services.gmail_service.build") as mock_build:
                mock_creds.side_effect = AssertionError("Should not load credentials in dry-run")
                mock_build.side_effect = AssertionError("Should not build service in dry-run")

                result = gs.send_email(payload)

                self.assertTrue(result.get("dry_run"))
                self.assertIsNone(result.get("message_id"))
                self.assertEqual(result.get("to"), "lead@example.com")
                self.assertEqual(result.get("subject"), "First Line Second Line")
                self.assertIn("DRY RUN", result.get("message", ""))

    # -----------------------------------------------------------------------
    # D. Live success
    # -----------------------------------------------------------------------

    def test_live_send_email_success(self) -> None:
        """Live send sets explicit 10s timeout, uses http transport, and preserves unicode."""
        import services.gmail_service as gs
        from models import EmailPayload

        payload = EmailPayload(
            to_email="recipient@example.com",
            to_name="Recipient Name",
            subject="Property\nTour Request",
            body="Special characters: em dash \u2014 and accented \u00e9.",
            lead_score=90,
        )

        live_env = {
            "GMAIL_DRY_RUN": "false",
            "GMAIL_SENDER_EMAIL": "agent@example.com",
        }

        with patch.dict(os.environ, live_env, clear=True):
            with patch("services.gmail_service._load_credentials") as mock_load_creds, \
                 patch("services.gmail_service.httplib2.Http") as mock_http_cls, \
                 patch("services.gmail_service.google_auth_httplib2.AuthorizedHttp") as mock_auth_cls, \
                 patch("services.gmail_service.build") as mock_build:

                mock_creds = MagicMock()
                mock_load_creds.return_value = mock_creds

                mock_http_inst = MagicMock()
                mock_http_cls.return_value = mock_http_inst

                mock_auth_inst = MagicMock()
                mock_auth_cls.return_value = mock_auth_inst

                mock_service = MagicMock()
                mock_build.return_value = mock_service
                mock_send_req = MagicMock()
                mock_send_req.execute.return_value = {"id": "abc123"}
                mock_service.users().messages().send.return_value = mock_send_req

                result = gs.send_email(payload)

                # 1. Http client timeout check
                mock_http_cls.assert_called_once_with(timeout=10)

                # 2. build() called with http= and not credentials=
                mock_build.assert_called_once()
                build_args, build_kwargs = mock_build.call_args
                self.assertEqual(build_kwargs.get("http"), mock_auth_inst)
                self.assertNotIn("credentials", build_kwargs)

                # 3. Confirmation result
                self.assertEqual(result.get("message_id"), "abc123")
                self.assertFalse(result.get("dry_run"))
                self.assertEqual(result.get("to"), "recipient@example.com")
                self.assertEqual(result.get("subject"), "Property Tour Request")

                # 4. MIME message decoding and verification
                send_mock = mock_service.users().messages().send
                send_mock.assert_called_once()
                send_args, send_kwargs = send_mock.call_args
                self.assertEqual(send_kwargs.get("userId"), "me")

                raw_b64 = send_kwargs.get("body", {}).get("raw", "")
                raw_bytes = base64.urlsafe_b64decode(raw_b64)
                parsed_msg = email.message_from_bytes(raw_bytes)

                self.assertEqual(parsed_msg["To"], "recipient@example.com")
                self.assertEqual(parsed_msg["From"], "agent@example.com")
                self.assertEqual(parsed_msg["Subject"], "Property Tour Request")

                body_content = (
                    parsed_msg.get_payload(0).get_payload(decode=True).decode("utf-8")
                )
                self.assertIn("\u2014", body_content)
                self.assertIn("\u00e9", body_content)
                self.assertEqual(body_content, payload.body)

    # -----------------------------------------------------------------------
    # E. Missing environment & credentials
    # -----------------------------------------------------------------------

    def test_missing_sender_email_raises_config_error(self) -> None:
        """Live mode with missing GMAIL_SENDER_EMAIL raises EmailSendError(category='config')."""
        import services.gmail_service as gs
        from services.gmail_service import EmailSendError
        from models import EmailPayload

        payload = EmailPayload(
            to_email="test@example.com",
            to_name="Test",
            subject="Subject",
            body="Body",
            lead_score=50,
        )

        with patch.dict(os.environ, {"GMAIL_DRY_RUN": "false"}, clear=True):
            with self.assertRaises(EmailSendError) as ctx:
                gs.send_email(payload)
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("GMAIL_SENDER_EMAIL", str(ctx.exception))

    def test_missing_token_json_raises_config_error(self) -> None:
        """Missing token.json raises EmailSendError(category='config') referencing GET /email/auth."""
        import services.gmail_service as gs
        from services.gmail_service import EmailSendError
        from models import EmailPayload

        payload = EmailPayload(
            to_email="test@example.com",
            to_name="Test",
            subject="Subject",
            body="Body",
            lead_score=50,
        )

        env = {
            "GMAIL_DRY_RUN": "false",
            "GMAIL_SENDER_EMAIL": "sender@example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("services.gmail_service.TOKEN_PATH", "/non/existent/token.json"):
                with self.assertRaises(EmailSendError) as ctx:
                    gs.send_email(payload)
                self.assertEqual(ctx.exception.category, "config")
                self.assertIn("GET /email/auth", str(ctx.exception))

    # -----------------------------------------------------------------------
    # F & G. Error classification & privacy preservation
    # -----------------------------------------------------------------------

    def test_error_classification_and_safety(self) -> None:
        """Classify errors correctly while guaranteeing zero leakage of recipient, subject, or body."""
        import services.gmail_service as gs
        from services.gmail_service import EmailSendError
        from models import EmailPayload

        payload = EmailPayload(
            to_email="confidential.lead@example.com",
            to_name="Lead Name",
            subject="Confidential Subject Line",
            body="Confidential Offer Body Text",
            lead_score=95,
        )

        def _json_err(reason: str) -> bytes:
            return json.dumps(
                {"error": {"errors": [{"reason": reason}]}}
            ).encode("utf-8")

        test_cases = [
            ("400 -> recipient", _make_http_error(400), "recipient", None),
            ("401 -> config", _make_http_error(401), "config", None),
            (
                "403 userRateLimitExceeded -> transient",
                _make_http_error(403, _json_err("userRateLimitExceeded")),
                "transient",
                None,
            ),
            (
                "403 rateLimitExceeded -> transient",
                _make_http_error(403, _json_err("rateLimitExceeded")),
                "transient",
                None,
            ),
            (
                "403 accessDenied -> config",
                _make_http_error(403, _json_err("accessDenied")),
                "config",
                None,
            ),
            ("404 -> config", _make_http_error(404), "config", None),
            ("429 -> transient", _make_http_error(429), "transient", None),
            ("500 -> transient", _make_http_error(500), "transient", None),
            ("503 -> transient", _make_http_error(503), "transient", None),
            (
                "ServerNotFoundError -> transient",
                httplib2.ServerNotFoundError("Unable to resolve host"),
                "transient",
                None,
            ),
            (
                "TimeoutError -> transient",
                TimeoutError("Connection timed out"),
                "transient",
                None,
            ),
            (
                "ConnectionError -> transient",
                ConnectionError("Connection reset by peer"),
                "transient",
                None,
            ),
            (
                "SSLError -> transient",
                ssl.SSLError("Handshake failure"),
                "transient",
                None,
            ),
            (
                "TransportError -> transient",
                TransportError("Network transport failed"),
                "transient",
                None,
            ),
            (
                "RefreshError -> config",
                RefreshError("Token revoked by user"),
                "config",
                None,
            ),
            (
                "ValueError -> config with name",
                ValueError("Internal unexpected parsing problem"),
                "config",
                "ValueError",
            ),
        ]

        live_env = {
            "GMAIL_DRY_RUN": "false",
            "GMAIL_SENDER_EMAIL": "agent@example.com",
        }

        for label, error_to_raise, expected_category, expected_text in test_cases:
            with self.subTest(case=label):
                with patch.dict(os.environ, live_env, clear=True):
                    with patch("services.gmail_service._load_credentials"), \
                         patch("services.gmail_service.httplib2.Http"), \
                         patch("services.gmail_service.google_auth_httplib2.AuthorizedHttp"), \
                         patch("services.gmail_service.build") as mock_build:

                        mock_service = MagicMock()
                        mock_build.return_value = mock_service
                        mock_service.users().messages().send().execute.side_effect = (
                            error_to_raise
                        )

                        with self.assertRaises(EmailSendError) as ctx:
                            gs.send_email(payload)

                        exc = ctx.exception
                        # Category validation
                        self.assertEqual(exc.category, expected_category)
                        # Inherits from RuntimeError
                        self.assertIsInstance(exc, RuntimeError)

                        error_str = str(exc)
                        # Privacy verification (never leak recipient, subject, or body)
                        self.assertNotIn(payload.to_email, error_str)
                        self.assertNotIn(payload.subject, error_str)
                        self.assertNotIn(payload.body, error_str)

                        if expected_text:
                            self.assertIn(expected_text, error_str)


if __name__ == "__main__":
    unittest.main()
