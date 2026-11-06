"""
tests/test_sms_router.py

Integration tests for the /sms/send HTTP endpoint (routers/sms.py).

Rules enforced:
- main.py is never imported.
- A throwaway FastAPI app is built per test using only the SMS router.
- No network access; .env is never read.
- env vars injected via patch.dict(os.environ, ..., clear=True).
- send_sms is patched at the router's import location when needed.
"""

import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_client(env: dict | None = None) -> TestClient:
    """
    Reload modules under *env* and return a TestClient.

    The reload is isolated, but the caller is responsible for keeping
    patch.dict active while the client makes requests (env vars are read
    at request time, not at reload time).
    """
    env = env or {}
    with patch.dict(os.environ, env, clear=True):
        # Reload the service first (router imports from it)
        import services.twilio_service as svc
        importlib.reload(svc)

        import routers.sms as sms_router
        importlib.reload(sms_router)

        app = FastAPI()
        app.include_router(sms_router.router)
        return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Real dry-run path (env cleared, send_sms NOT patched)
# ---------------------------------------------------------------------------
class TestDryRunRealPath(unittest.TestCase):

    def setUp(self):
        # Keep TWILIO_DRY_RUN absent (clear=True) for the entire test,
        # including the HTTP request.  Without this, a shell-level
        # TWILIO_DRY_RUN=false would trigger live mode and return 500.
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    def test_valid_payload_returns_200_dry_run(self):
        """
        With no TWILIO_DRY_RUN set (defaults to dry-run), a valid payload
        must return 200 with dry_run=true and the normalised E.164 'to' field.
        """
        client = _build_client(env={})

        resp = client.post(
            "/sms/send",
            json={
                "to_phone": "+1-415-555-0192",
                "message": (
                    "Hi Sarah, an agent will contact you shortly. "
                    "Reply STOP to opt out."
                ),
                "lead_score": 87,
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["dry_run"])
        self.assertEqual(body["to"], "+14155550192")


# ---------------------------------------------------------------------------
# 2. Invalid to_phone in real path
# ---------------------------------------------------------------------------
class TestInvalidPhoneRealPath(unittest.TestCase):

    def setUp(self):
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    def test_national_format_phone_returns_422(self):
        """
        'to_phone' without country code is not valid E.164; the service
        raises ValueError which the router maps to 422.
        """
        client = _build_client(env={})

        resp = client.post(
            "/sms/send",
            json={
                "to_phone": "415-555-0192",
                "message": "Hello there.",
                "lead_score": 50,
            },
        )
        self.assertEqual(resp.status_code, 422)
        detail = resp.json().get("detail", "")
        self.assertIn("international format", str(detail))


# ---------------------------------------------------------------------------
# 3. Patched send_sms raising various errors
# ---------------------------------------------------------------------------
class TestPatchedSendSmsErrors(unittest.TestCase):

    def _post(self, make_exc, env=None):
        """
        POST to /sms/send with send_sms patched to raise an exception.

        `make_exc` is a *callable* that accepts the reloaded SMSSendError class
        and returns the exception to raise. This avoids class-identity mismatches
        that occur when importlib.reload() creates a new class object while the
        endpoint's except-clause still holds a reference to the old one.
        """
        env = env or {}

        with patch.dict(os.environ, env, clear=True):
            import services.twilio_service as svc
            importlib.reload(svc)
            import routers.sms as sms_router
            importlib.reload(sms_router)

            # Obtain SMSSendError from the *reloaded* module so that isinstance
            # checks inside the endpoint's except-clauses match correctly.
            ReloadedSMSSendError = svc.SMSSendError
            exc = make_exc(ReloadedSMSSendError)

            app = FastAPI()
            app.include_router(sms_router.router)
            client = TestClient(app, raise_server_exceptions=False)

            with patch("routers.sms.send_sms", side_effect=exc):
                resp = client.post(
                    "/sms/send",
                    json={
                        "to_phone": "+14155550192",
                        "message": "Test message.",
                        "lead_score": 60,
                    },
                )
        return resp

    def test_sms_send_error_recipient_returns_422(self):
        resp = self._post(lambda E: E("bad number", category="recipient"))
        self.assertEqual(resp.status_code, 422)

    def test_sms_send_error_transient_returns_502(self):
        resp = self._post(lambda E: E("transient failure", category="transient"))
        self.assertEqual(resp.status_code, 502)

    def test_sms_send_error_config_returns_500(self):
        resp = self._post(lambda E: E("config problem", category="config"))
        self.assertEqual(resp.status_code, 500)

    def test_value_error_returns_422(self):
        resp = self._post(lambda _E: ValueError("phone is bad"))
        self.assertEqual(resp.status_code, 422)

    def test_unexpected_runtime_error_returns_500_generic_detail(self):
        resp = self._post(lambda _E: RuntimeError("boom"))
        self.assertEqual(resp.status_code, 500)
        detail = resp.json().get("detail", "")
        # The word "boom" must NOT appear in the response
        self.assertNotIn("boom", str(detail))
        # Must be the generic message from the router
        self.assertIn("Unexpected error", str(detail))


# ---------------------------------------------------------------------------
# 4. Message longer than 160 characters -> 422 (Pydantic validation)
# ---------------------------------------------------------------------------
class TestMessageTooLong(unittest.TestCase):

    def setUp(self):
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    def test_message_over_160_chars_returns_422(self):
        """
        SMSPayload.message has a max 160-char validator; Pydantic rejects it
        before the endpoint body even executes.
        """
        client = _build_client(env={})

        resp = client.post(
            "/sms/send",
            json={
                "to_phone": "+14155550192",
                "message": "a" * 161,
                "lead_score": 50,
            },
        )
        self.assertEqual(resp.status_code, 422)


# ---------------------------------------------------------------------------
# 5. OpenAPI schema contains /sms/send
# ---------------------------------------------------------------------------
class TestOpenAPISchema(unittest.TestCase):

    def setUp(self):
        self._env_patcher = patch.dict(os.environ, {}, clear=True)
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

    def test_openapi_schema_has_sms_send_path(self):
        client = _build_client(env={})
        resp = client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()
        self.assertIn("/sms/send", schema.get("paths", {}))


if __name__ == "__main__":
    unittest.main()
