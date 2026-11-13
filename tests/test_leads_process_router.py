"""
tests/test_leads_process_router.py

Unit tests for POST /leads/process and related router contracts in routers/leads.py.
Follows all Phase 9 rules:
- Uses ONLY unittest, unittest.mock, and fastapi.testclient.TestClient.
- main.py is NEVER imported.
- Uses a throwaway FastAPI app containing only leads.router.
- No network access, no .env file access.
- Environment-dependent tests use patch.dict(os.environ, {...}, clear=True).
- Uses subTest for table-driven scenarios.
"""

import inspect
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from models import LeadInput
import routers.leads as leads_router
from services.pipeline_service import PipelineResult, StepResult


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def _build_test_client() -> TestClient:
    """Build a throwaway FastAPI app with only the leads router."""
    app = FastAPI()
    app.include_router(leads_router.router)
    return TestClient(app, raise_server_exceptions=False)


def _build_sample_pipeline_result(
    status: str = "completed",
    lead_id: str = "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    retry_steps: list[str] | None = None,
) -> PipelineResult:
    """Construct a full PipelineResult fixture."""
    steps = [
        StepResult(name="intake", status="succeeded", reason="normalized", duration_ms=2),
        StepResult(name="score", status="succeeded", reason="scored", duration_ms=10),
        StepResult(name="store", status="succeeded", reason="created", duration_ms=5),
        StepResult(name="email", status="succeeded" if status != "failed" else "failed",
                   reason="sent" if status != "failed" else "transient",
                   external_id="m1", flag_recorded=True, duration_ms=15),
        StepResult(name="sms", status="succeeded", reason="sent",
                   external_id="SM1", flag_recorded=True, duration_ms=12),
        StepResult(name="crm", status="succeeded", reason="synced",
                   external_id="512", flag_recorded=True, duration_ms=20),
    ]
    return PipelineResult(
        lead_id=lead_id,
        status=status,
        persisted=True if status != "failed" else False,
        created=True if status != "failed" else None,
        idempotency_key_generated=False,
        category="hot",
        score=87,
        steps=steps,
        retry_steps=retry_steps if retry_steps is not None else [],
        repair=[],
        failed_step=None if status != "failed" else "email",
    )


def _valid_request_payload() -> dict:
    """Build a valid JSON request payload for POST /leads/process."""
    return {
        "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "lead": {
            "name": "Sarah Mitchell",
            "email": "sarah.mitchell@example.com",
            "phone": "+1-415-555-0192",
            "message": "Looking for a 3-bedroom house in Austin.",
            "property_type": "house",
            "preferred_location": "Austin, TX",
            "budget": "$450k-$550k",
            "timeline": "within 3 months",
        },
    }


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

class TestProcessLeadEndpointResponses(unittest.TestCase):
    """1. Status code mapping for completed, partial, and failed outcomes."""

    def setUp(self):
        self.client = _build_test_client()
        self.payload = _valid_request_payload()

    def test_process_lead_completed_returns_200_with_six_steps(self):
        result = _build_sample_pipeline_result(status="completed")
        with patch("routers.leads.process_lead", return_value=result):
            resp = self.client.post("/leads/process", json=self.payload)

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(len(body["steps"]), 6)
        self.assertEqual(body["lead_id"], result.lead_id)
        self.assertFalse(body["retry"]["recommended"])

    def test_process_lead_partial_returns_200(self):
        result = _build_sample_pipeline_result(status="partial")
        with patch("routers.leads.process_lead", return_value=result):
            resp = self.client.post("/leads/process", json=self.payload)

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "partial")
        self.assertEqual(len(body["steps"]), 6)
        self.assertEqual(body["lead_id"], result.lead_id)

    def test_process_lead_failed_with_retry_steps_returns_502(self):
        result = _build_sample_pipeline_result(status="failed", retry_steps=["score"])
        with patch("routers.leads.process_lead", return_value=result):
            resp = self.client.post("/leads/process", json=self.payload)

        self.assertEqual(resp.status_code, 502)
        body = resp.json()
        self.assertEqual(body["status"], "failed")
        self.assertTrue(body["retry"]["recommended"])
        self.assertEqual(body["retry"]["steps"], ["score"])
        self.assertEqual(len(body["steps"]), 6)

    def test_process_lead_failed_without_retry_steps_returns_500(self):
        result = _build_sample_pipeline_result(status="failed", retry_steps=[])
        with patch("routers.leads.process_lead", return_value=result):
            resp = self.client.post("/leads/process", json=self.payload)

        self.assertEqual(resp.status_code, 500)
        body = resp.json()
        self.assertEqual(body["status"], "failed")
        self.assertFalse(body["retry"]["recommended"])
        self.assertEqual(body["retry"]["steps"], [])
        self.assertEqual(len(body["steps"]), 6)


class TestProcessLeadExceptions(unittest.TestCase):
    """2. Handling of exceptions raised by process_lead."""

    def setUp(self):
        self.client = _build_test_client()
        self.payload = _valid_request_payload()

    def test_process_lead_raising_value_error_returns_422(self):
        with patch("routers.leads.process_lead", side_effect=ValueError("Invalid UUID or argument")):
            resp = self.client.post("/leads/process", json=self.payload)

        self.assertEqual(resp.status_code, 422)
        body = resp.json()
        self.assertIn("Invalid UUID or argument", body.get("detail", ""))

    def test_process_lead_raising_generic_exception_returns_500_with_generic_detail(self):
        with patch("routers.leads.process_lead", side_effect=Exception("boom")):
            resp = self.client.post("/leads/process", json=self.payload)

        self.assertEqual(resp.status_code, 500)
        body = resp.json()
        detail = body.get("detail", "")
        self.assertNotIn("boom", detail)
        self.assertEqual(detail, "Unexpected error while processing the lead.")


class TestProcessLeadValidation(unittest.TestCase):
    """3. Fast-fail schema and UUID validation before collaborator execution."""

    def setUp(self):
        self.client = _build_test_client()

    def test_invalid_uuid_returns_422_without_calling_process_lead(self):
        payload = _valid_request_payload()
        payload["id"] = "abc"

        with patch("routers.leads.process_lead") as mock_process:
            resp = self.client.post("/leads/process", json=payload)

        self.assertEqual(resp.status_code, 422)
        mock_process.assert_not_called()

    def test_missing_lead_field_returns_422_without_calling_process_lead(self):
        payload = {
            "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "lead": {
                "name": "Sarah Mitchell",
                # missing email, phone, message
            },
        }

        with patch("routers.leads.process_lead") as mock_process:
            resp = self.client.post("/leads/process", json=payload)

        self.assertEqual(resp.status_code, 422)
        mock_process.assert_not_called()


class TestProcessLeadUnpatchedRealEnvCleared(unittest.TestCase):
    """4. Unpatched execution in an empty environment fails score step."""

    def test_unpatched_real_path_cleared_env_score_fails_config_returns_500(self):
        client = _build_test_client()
        payload = _valid_request_payload()

        with patch.dict(os.environ, {}, clear=True):
            resp = client.post("/leads/process", json=payload)

        self.assertEqual(resp.status_code, 500)
        body = resp.json()

        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["failed_step"], "score")
        self.assertFalse(body["persisted"])

        steps = body["steps"]
        step_dict = {s["name"]: s for s in steps}

        self.assertEqual(step_dict["score"]["status"], "failed")
        self.assertEqual(step_dict["score"]["reason"], "config")
        self.assertIn("OPENROUTER_API_KEY", step_dict["score"]["error"]["message"])

        for name in ("store", "email", "sms", "crm"):
            self.assertEqual(step_dict[name]["status"], "skipped")
            self.assertEqual(step_dict[name]["reason"], "upstream_failed")


class TestLeadsRouterRegressionAndContract(unittest.TestCase):
    """5. Regression check for existing endpoints and OpenAPI schema contracts."""

    def setUp(self):
        self.client = _build_test_client()

    def test_get_leads_test_returns_original_message(self):
        resp = self.client.get("/leads/test")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"message": "Leads router is working correctly"})

    def test_openapi_schema_contains_all_leads_endpoints(self):
        app = FastAPI()
        app.include_router(leads_router.router)
        paths = app.openapi().get("paths", {})

        self.assertIn("/leads/store", paths)
        self.assertIn("/leads/{lead_id}", paths)
        self.assertIn("/leads/{lead_id}/flags", paths)
        self.assertIn("/leads/process", paths)

    def test_process_lead_endpoint_is_not_coroutine(self):
        self.assertFalse(inspect.iscoroutinefunction(leads_router.process_lead_endpoint))


if __name__ == "__main__":
    unittest.main()
