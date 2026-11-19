"""
tests/test_main_auth.py

Integration tests for API key authentication and default-deny protection
wired into the main FastAPI application (main.py).

Rules enforced:
- main is imported with dotenv.load_dotenv patched before import and
  environment patched with patch.dict(clear=True) and a 40-character API_SECRET_KEY.
- main is dropped from sys.modules first to ensure fresh initialization.
- Only unittest, unittest.mock, and fastapi.testclient.TestClient.
- Pure ASCII and saved as UTF-8 without BOM.
- No __main__ entrypoint.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
import uuid
from unittest.mock import MagicMock, patch

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from services.security import PUBLIC_PATHS, make_oauth_state, verify_oauth_state

_SECRET_40 = "A" * 40


# ---------------------------------------------------------------------------
# Load main module with isolated environment
# ---------------------------------------------------------------------------

sys.modules.pop("main", None)
with patch("dotenv.load_dotenv"), patch.dict(
    os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True
):
    import main


class TestMainAuth(unittest.TestCase):
    """Integration test suite for authentication enforcement in main.py."""

    def setUp(self) -> None:
        """Initialize the FastAPI test client with the 40-char secret configured."""
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.headers = {"X-API-Key": _SECRET_40}

    # -----------------------------------------------------------------------
    # (a) Default-deny walk across all routes
    # -----------------------------------------------------------------------

    def test_default_deny_walk_all_routes_without_key(self) -> None:
        """Every APIRoute requires an API key (returns 401) unless path is in PUBLIC_PATHS."""
        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            for route in main.app.routes:
                if not isinstance(route, APIRoute):
                    continue

                # Substitute path parameters (e.g. {lead_id}) with a valid UUID
                resolved_path = re.sub(r"\{[^}]+\}", str(uuid.uuid4()), route.path)
                # Select a non-HEAD HTTP method
                method = next(m.lower() for m in route.methods if m != "HEAD")
                caller = getattr(self.client, method)

                resp = caller(resolved_path)

                with self.subTest(route=route.path, method=method):
                    if route.path in PUBLIC_PATHS:
                        self.assertNotEqual(
                            resp.status_code,
                            401,
                            f"Public route {route.path!r} must not return 401 without API key",
                        )
                    else:
                        self.assertEqual(
                            resp.status_code,
                            401,
                            f"Protected route {route.path!r} must return 401 without API key",
                        )

    # -----------------------------------------------------------------------
    # (b) Protected endpoints with valid key
    # -----------------------------------------------------------------------

    def test_protected_endpoints_with_valid_key(self) -> None:
        """GET /leads/test returns 200; empty-body POSTs return 422 indicating auth passed."""
        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            # GET /leads/test returns 200 with valid key
            resp_leads = self.client.get("/leads/test", headers=self.headers)
            self.assertEqual(
                resp_leads.status_code,
                200,
                "GET /leads/test must return 200 with valid API key",
            )

            # Empty-body POST requests return 422 Unprocessable Entity
            # (confirms authentication succeeded and Pydantic validation rejected empty payload)
            empty_post_routes = [
                "/leads/process",
                "/email/send",
                "/sms/send",
                "/crm/contact",
                "/ai/score-lead",
            ]
            for path in empty_post_routes:
                with self.subTest(endpoint=path):
                    resp = self.client.post(path, headers=self.headers, json={})
                    self.assertEqual(
                        resp.status_code,
                        422,
                        f"POST {path} with empty body must return 422 once authenticated",
                    )

    # -----------------------------------------------------------------------
    # (c) Public endpoints accessible without key
    # -----------------------------------------------------------------------

    def test_public_routes_return_200_without_key(self) -> None:
        """Endpoints /health and / return 200 without providing an API key."""
        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            for path in ["/health", "/"]:
                with self.subTest(path=path):
                    resp = self.client.get(path)
                    self.assertEqual(
                        resp.status_code,
                        200,
                        f"Public endpoint {path!r} must return 200 without an API key",
                    )

    # -----------------------------------------------------------------------
    # (d) Secret unset behavior
    # -----------------------------------------------------------------------

    def test_secret_unset_gives_503_for_protected_and_200_for_public(self) -> None:
        """When API_SECRET_KEY is unset, protected routes return 503 and public return 200."""
        with patch.dict(os.environ, {}, clear=True):
            # Public routes remain healthy
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.get("/").status_code, 200)

            # Protected routes fail closed with 503 Service Unavailable
            resp_prot = self.client.get("/leads/test")
            self.assertEqual(resp_prot.status_code, 503)
            self.assertEqual(
                resp_prot.json().get("detail"),
                "API authentication is not configured.",
            )

    # -----------------------------------------------------------------------
    # (e) /email/auth endpoint behavior
    # -----------------------------------------------------------------------

    def test_email_auth_without_key_401_and_with_key_verifies_state(self) -> None:
        """GET /email/auth without key returns 401; with key receives valid OAuth state."""
        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            # Unauthenticated request -> 401
            resp_unauth = self.client.get("/email/auth")
            self.assertEqual(resp_unauth.status_code, 401)

            # Authenticated request with mocked get_auth_url
            with patch("routers.email.get_auth_url") as mock_get_auth:
                mock_get_auth.return_value = "https://accounts.google.com/o/oauth2/auth?test=1"
                resp_auth = self.client.get("/email/auth", headers=self.headers)
                self.assertEqual(resp_auth.status_code, 200)
                mock_get_auth.assert_called_once()

                # Verify state passed to get_auth_url is authentic and valid
                passed_state = mock_get_auth.call_args[0][0]
                self.assertTrue(
                    verify_oauth_state(passed_state),
                    "State passed to get_auth_url must pass verify_oauth_state",
                )

    # -----------------------------------------------------------------------
    # (f) /email/auth/callback validation matrix
    # -----------------------------------------------------------------------

    def test_email_auth_callback_validation_matrix(self) -> None:
        """Verify callback handles errors, missing code, invalid state, and valid exchange."""
        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            # 1. ?error=... -> 400
            resp_error = self.client.get("/email/auth/callback?error=access_denied")
            self.assertEqual(resp_error.status_code, 400)
            self.assertIn("Authorization denied", resp_error.json().get("detail", ""))

            # 2. No params -> 400
            resp_no_params = self.client.get("/email/auth/callback")
            self.assertEqual(resp_no_params.status_code, 400)
            self.assertIn("Missing 'code'", resp_no_params.json().get("detail", ""))

            # 3. Code present but missing state -> 400, exchange_code never called
            with patch("routers.email.exchange_code") as mock_exchange:
                resp_missing_state = self.client.get("/email/auth/callback?code=mock_code_123")
                self.assertEqual(resp_missing_state.status_code, 400)
                self.assertIn(
                    "Invalid or expired OAuth state",
                    resp_missing_state.json().get("detail", ""),
                )
                mock_exchange.assert_not_called()

            # 4. Code present but invalid state -> 400, exchange_code never called
            with patch("routers.email.exchange_code") as mock_exchange:
                resp_bad_state = self.client.get(
                    "/email/auth/callback?code=mock_code_123&state=bad.state.signature"
                )
                self.assertEqual(resp_bad_state.status_code, 400)
                self.assertIn(
                    "Invalid or expired OAuth state",
                    resp_bad_state.json().get("detail", ""),
                )
                mock_exchange.assert_not_called()

            # 5. Code present + valid state -> 200, exchange_code called once
            valid_state = make_oauth_state()
            with patch("routers.email.exchange_code") as mock_exchange:
                mock_exchange.return_value = None
                resp_success = self.client.get(
                    f"/email/auth/callback?code=mock_code_123&state={valid_state}"
                )
                self.assertEqual(resp_success.status_code, 200)
                mock_exchange.assert_called_once_with("mock_code_123")
                self.assertIn("Authorization successful", resp_success.json().get("message", ""))
