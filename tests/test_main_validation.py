"""
tests/test_main_validation.py

Integration tests for the sanitized RequestValidationError (422) handler
registered in main.py.

Rules enforced:
- main is imported with dotenv.load_dotenv patched before import and
  environment patched with patch.dict(clear=True) and a 40-character API_SECRET_KEY.
- main is dropped from sys.modules first to ensure fresh initialization.
- Only stdlib, unittest, unittest.mock, and fastapi.testclient.TestClient.
- Pure ASCII and saved as UTF-8 without BOM with LF line endings.
- No __main__ entrypoint.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

_SECRET_40 = "A" * 40
_ALLOWED_ERROR_KEYS = {"loc", "msg", "type"}


# ---------------------------------------------------------------------------
# Load main module with isolated environment
# ---------------------------------------------------------------------------

sys.modules.pop("main", None)
with patch("dotenv.load_dotenv"), patch.dict(
    os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True
):
    import main


class TestMainValidation(unittest.TestCase):
    """Integration tests for sanitized 422 validation responses in main.py."""

    def setUp(self) -> None:
        """Initialize the FastAPI test client and default auth headers."""
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.headers = {"X-API-Key": _SECRET_40}

    def _assert_sanitized_422(self, resp, canaries: tuple[str, ...]) -> None:
        """Verify 422 structure, error key whitelist, and absence of canary strings."""
        self.assertEqual(resp.status_code, 422)
        body = resp.json()
        self.assertIsInstance(body, dict)
        self.assertEqual(set(body.keys()), {"detail"})
        detail = body["detail"]
        self.assertIsInstance(detail, list)
        self.assertGreater(len(detail), 0)
        for item in detail:
            self.assertIsInstance(item, dict)
            self.assertTrue(
                set(item.keys()).issubset(_ALLOWED_ERROR_KEYS),
                f"Unexpected validation error keys: {set(item.keys())}",
            )
        raw_text = resp.text
        for canary in canaries:
            self.assertNotIn(
                canary,
                raw_text,
                f"Canary string {canary!r} leaked in 422 response body",
            )

    def test_422_validation_cases_strip_input_ctx_and_url(self) -> None:
        """All 422 validation failures return only loc/msg/type without echoing canaries."""
        cases = [
            {
                "name": "leads_process_missing_nested_email",
                "method": "post",
                "path": "/leads/process",
                "kwargs": {
                    "json": {
                        "lead": {
                            "name": "Zed Canary",
                            "phone": "+14155550192",
                            "message": "canary message text",
                        }
                    }
                },
                "canaries": ("Zed Canary", "+14155550192", "canary message text"),
            },
            {
                "name": "leads_process_whitespace_name_value_error",
                "method": "post",
                "path": "/leads/process",
                "kwargs": {
                    "json": {
                        "lead": {
                            "name": "   ",
                            "email": "canary.lead@example.com",
                            "phone": "+14155550192",
                            "message": "canary message text",
                        }
                    }
                },
                "canaries": ("canary.lead@example.com", "+14155550192", "canary message text"),
            },
            {
                "name": "crm_contact_invalid_lead_status",
                "method": "post",
                "path": "/crm/contact",
                "kwargs": {
                    "json": {
                        "email": "canary.lead@example.com",
                        "firstname": "Alice",
                        "lastname": "Smith",
                        "phone": "+14155550192",
                        "lead_score": 75,
                        "lead_status": "lukewarm",
                    }
                },
                "canaries": ("canary.lead@example.com",),
            },
            {
                "name": "email_send_empty_subject",
                "method": "post",
                "path": "/email/send",
                "kwargs": {
                    "json": {
                        "to_email": "canary.lead@example.com",
                        "to_name": "Alice Smith",
                        "subject": "",
                        "body": "Hello there.",
                        "lead_score": 80,
                    }
                },
                "canaries": ("canary.lead@example.com",),
            },
            {
                "name": "captcha_verify_non_string_token",
                "method": "post",
                "path": "/captcha/verify",
                "kwargs": {
                    "json": {"token": ["CANARY-TOKEN"]},
                },
                "canaries": ("CANARY-TOKEN",),
            },
            {
                "name": "ai_score_lead_missing_message",
                "method": "post",
                "path": "/ai/score-lead",
                "kwargs": {
                    "json": {
                        "name": "Zed Canary",
                        "email": "valid.lead@example.com",
                        "phone": "+14155550192",
                    }
                },
                "canaries": ("Zed Canary", "+14155550192"),
            },
            {
                "name": "leads_get_invalid_uuid_path_param",
                "method": "get",
                "path": "/leads/abc",
                "kwargs": {},
                "canaries": (),
            },
            {
                "name": "leads_process_malformed_json_body",
                "method": "post",
                "path": "/leads/process",
                "kwargs": {
                    "content": b"{not json",
                    "extra_headers": {"Content-Type": "application/json"},
                },
                "canaries": ("{not json",),
            },
        ]

        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            for case in cases:
                with self.subTest(case=case["name"]):
                    kwargs = dict(case["kwargs"])
                    extra_headers = kwargs.pop("extra_headers", {})
                    headers = {**self.headers, **extra_headers}
                    caller = getattr(self.client, case["method"])
                    resp = caller(case["path"], headers=headers, **kwargs)
                    self._assert_sanitized_422(resp, case["canaries"])

    def test_auth_and_routing_precedence_preserved(self) -> None:
        """Auth checks, /leads/test 200, and 404 responses behave normally."""
        with patch.dict(os.environ, {"API_SECRET_KEY": _SECRET_40}, clear=True):
            # Valid JSON with invalid fields and no API key -> 401 before validation
            resp_no_key = self.client.post(
                "/leads/process",
                json={"lead": {"name": "   "}},
            )
            self.assertEqual(resp_no_key.status_code, 401)

            # Authenticated GET /leads/test still returns 200
            resp_test = self.client.get("/leads/test", headers=self.headers)
            self.assertEqual(resp_test.status_code, 200)

            # Unknown path with key returns 404 with normal body
            resp_404 = self.client.get("/does-not-exist", headers=self.headers)
            self.assertEqual(resp_404.status_code, 404)
            self.assertEqual(resp_404.json(), {"detail": "Not Found"})

        # API_SECRET_KEY unset -> 503 with configuration detail
        with patch.dict(os.environ, {}, clear=True):
            resp_503 = self.client.post(
                "/leads/process",
                headers=self.headers,
                json={"lead": {"name": "   "}},
            )
            self.assertEqual(resp_503.status_code, 503)
            self.assertEqual(
                resp_503.json(),
                {"detail": "API authentication is not configured."},
            )
