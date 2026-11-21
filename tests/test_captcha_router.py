"""
tests/test_captcha_router.py

Unit tests for the /captcha/verify HTTP endpoint (routers/captcha.py).

Rules enforced:
- Only unittest, unittest.mock, and fastapi.testclient.TestClient.
- No network access, no .env file read.
- Pure ASCII and saved as UTF-8 without BOM.
- Router tests build a throwaway FastAPI app: app = FastAPI(); app.include_router(captcha.router).
- Table-driven tests use subTest.
"""

from __future__ import annotations

import inspect
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.captcha as captcha
from services.captcha_service import CaptchaError


def _build_test_client() -> TestClient:
    """Build a throwaway FastAPI test client with only the captcha router included."""
    app = FastAPI()
    app.include_router(captcha.router)
    return TestClient(app, raise_server_exceptions=False)


class TestCaptchaRouter(unittest.TestCase):
    """Test suite for the CAPTCHA router HTTP endpoints."""

    def setUp(self) -> None:
        self.client = _build_test_client()

    def test_verified_result_passes_through_as_200(self) -> None:
        """Verified result returns 200 with verified=True and reason='ok'."""
        mock_result = {"verified": True, "reason": "ok", "error_codes": []}
        with patch("routers.captcha.verify_token", return_value=mock_result) as mock_verify:
            resp = self.client.post(
                "/captcha/verify",
                json={"token": "valid-token", "remote_ip": "1.2.3.4"},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["verified"])
            self.assertEqual(data["reason"], "ok")
            self.assertEqual(data["error_codes"], [])
            mock_verify.assert_called_once_with("valid-token", "1.2.3.4")

    def test_rejected_result_returns_200_with_verified_false(self) -> None:
        """Rejected result returns 200 with verified=False and provider reason."""
        mock_result = {
            "verified": False,
            "reason": "expired_or_used",
            "error_codes": ["timeout-or-duplicate"],
        }
        with patch("routers.captcha.verify_token", return_value=mock_result):
            resp = self.client.post("/captcha/verify", json={"token": "expired-token"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["verified"])
            self.assertEqual(data["reason"], "expired_or_used")
            self.assertEqual(data["error_codes"], ["timeout-or-duplicate"])

    def test_captcha_error_transient_502_config_500(self) -> None:
        """CaptchaError transient maps to 502, config maps to 500, both preserving detail."""
        cases = [
            ("transient", 502, "Provider temporarily unavailable."),
            ("config", 500, "Secret key rejected."),
        ]
        for cat, expected_status, msg in cases:
            with self.subTest(category=cat):
                with patch(
                    "routers.captcha.verify_token",
                    side_effect=CaptchaError(msg, category=cat),
                ):
                    resp = self.client.post("/captcha/verify", json={"token": "some-token"})
                    self.assertEqual(resp.status_code, expected_status)
                    self.assertEqual(resp.json().get("detail"), msg)

    def test_unexpected_runtime_error_returns_500_with_fixed_detail(self) -> None:
        """Unhandled exceptions return 500 with fixed detail and no leaked message."""
        with patch(
            "routers.captcha.verify_token",
            side_effect=RuntimeError("boom"),
        ):
            resp = self.client.post("/captcha/verify", json={"token": "some-token"})
            self.assertEqual(resp.status_code, 500)
            detail = resp.json().get("detail", "")
            self.assertEqual(detail, "Unexpected error while verifying the CAPTCHA.")
            self.assertNotIn("boom", detail)

    def test_empty_json_body_calls_verify_token_with_nones(self) -> None:
        """Empty JSON object {} calls verify_token with (None, None)."""
        mock_result = {"verified": False, "reason": "missing_token", "error_codes": []}
        with patch("routers.captcha.verify_token", return_value=mock_result) as mock_verify:
            resp = self.client.post("/captcha/verify", json={})
            self.assertEqual(resp.status_code, 200)
            mock_verify.assert_called_once_with(None, None)

    def test_non_string_token_returns_422_without_calling_service(self) -> None:
        """Non-string token (e.g. 123) produces 422 and never invokes verify_token."""
        with patch("routers.captcha.verify_token") as mock_verify:
            resp = self.client.post("/captcha/verify", json={"token": 123})
            self.assertEqual(resp.status_code, 422)
            mock_verify.assert_not_called()

    def test_openapi_schema_contains_required_status_codes(self) -> None:
        """OpenAPI schema contains /captcha/verify with documented 200, 422, 500, 502."""
        app = FastAPI()
        app.include_router(captcha.router)
        schema = app.openapi()
        paths = schema.get("paths", {})
        self.assertIn("/captcha/verify", paths)
        post_op = paths["/captcha/verify"].get("post", {})
        responses = post_op.get("responses", {})
        for code in ["200", "422", "500", "502"]:
            self.assertIn(code, responses, f"OpenAPI schema must document HTTP {code}")

    def test_handler_is_not_coroutine_function(self) -> None:
        """Endpoint handler is synchronous (def, not async def) to run in thread pool."""
        self.assertFalse(
            inspect.iscoroutinefunction(captcha.verify_captcha_endpoint),
            "verify_captcha_endpoint must be a regular def (not async def)",
        )

    def test_real_path_unconfigured_secret_returns_500_naming_env(self) -> None:
        """Unpatched real path with empty env fails closed returning 500 naming TURNSTILE_SECRET_KEY."""
        with patch.dict(os.environ, {}, clear=True):
            resp = self.client.post("/captcha/verify", json={"token": "test-token"})
            self.assertEqual(resp.status_code, 500)
            self.assertIn("TURNSTILE_SECRET_KEY", resp.json().get("detail", ""))

    def test_static_check_main_includes_captcha_router(self) -> None:
        """Verify main.py imports and includes captcha.router."""
        main_path = Path(__file__).resolve().parent.parent / "main.py"
        content = main_path.read_text(encoding="utf-8")
        self.assertRegex(
            content,
            r"from\s+routers\s+import\s+.*?\bcaptcha\b",
            "main.py must import captcha from routers",
        )
        self.assertRegex(
            content,
            r"app\.include_router\(\s*captcha\.router\s*\)",
            "main.py must include captcha.router",
        )
