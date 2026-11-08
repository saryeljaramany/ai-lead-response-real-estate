"""
tests/test_leads_router.py

Unit tests for routers/leads.py HTTP endpoints.

Rules enforced:
- main.py is never imported.
- Throwaway FastAPI app created with only the leads router.
- No network access; .env is never read.
- Env-dependent tests use patch.dict(os.environ, ..., clear=True).
- Service functions are patched in routers.leads.
"""

from datetime import datetime, timezone
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from models import LeadInput, LeadRecord, LeadScore
import routers.leads as leads_router
from services.supabase_service import LeadStoreError


def _build_client() -> TestClient:
    """Build a throwaway FastAPI app with the leads router."""
    app = FastAPI()
    app.include_router(leads_router.router)
    return TestClient(app, raise_server_exceptions=False)


def _build_sample_record(
    lead_id: str = "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    name: str = "Sarah Mitchell",
    email: str = "sarah.mitchell@example.com",
    phone: str = "+1-415-555-0192",
    message: str = "Looking for a 3-bedroom house in Austin.",
    property_type: str | None = "house",
    preferred_location: str | None = "Austin, TX",
    budget: str | None = "$450k-$550k",
    timeline: str | None = "within 3 months",
    score: int = 87,
    reasoning: str = "Clear budget and timeline.",
    suggested_action: str = "Contact today.",
    priority_level: str = "urgent",
    email_sent: bool = False,
    sms_sent: bool = False,
    hubspot_created: bool = False,
    created_at: datetime | None = None,
) -> LeadRecord:
    """Construct a full LeadRecord fixture."""
    return LeadRecord(
        id=lead_id,
        lead=LeadInput(
            name=name,
            email=email,
            phone=phone,
            message=message,
            property_type=property_type,
            preferred_location=preferred_location,
            budget=budget,
            timeline=timeline,
        ),
        score=LeadScore(
            score=score,
            reasoning=reasoning,
            suggested_action=suggested_action,
            priority_level=priority_level,
        ),
        email_sent=email_sent,
        sms_sent=sms_sent,
        hubspot_created=hubspot_created,
        created_at=created_at or datetime(2026, 4, 25, 21, 0, 0, tzinfo=timezone.utc),
    )


class TestLeadsRouterStore(unittest.TestCase):
    """Tests for POST /leads/store."""

    def setUp(self):
        self.client = _build_client()
        self.sample = _build_sample_record()
        self.payload = self.sample.model_dump(mode="json")

    def test_1_store_lead_success_returns_200(self):
        with patch("routers.leads.store_lead", return_value=(self.sample, True)):
            resp = self.client.post("/leads/store", json=self.payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["created"])
        self.assertEqual(body["record"]["id"], self.sample.id)
        self.assertEqual(body["record"]["lead"]["name"], self.sample.lead.name)

    def test_2_value_error_returns_422(self):
        with patch("routers.leads.store_lead", side_effect=ValueError("Invalid UUID or fields")):
            resp = self.client.post("/leads/store", json=self.payload)
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Invalid UUID or fields", resp.json().get("detail", ""))

    def test_2_lead_store_error_config_returns_500(self):
        exc = LeadStoreError("Schema misconfigured", category="config")
        with patch("routers.leads.store_lead", side_effect=exc):
            resp = self.client.post("/leads/store", json=self.payload)
        self.assertEqual(resp.status_code, 500)
        self.assertIn("Schema misconfigured", resp.json().get("detail", ""))

    def test_2_lead_store_error_transient_returns_502(self):
        exc = LeadStoreError("Database timeout", category="transient")
        with patch("routers.leads.store_lead", side_effect=exc):
            resp = self.client.post("/leads/store", json=self.payload)
        self.assertEqual(resp.status_code, 502)
        self.assertIn("Database timeout", resp.json().get("detail", ""))

    def test_2_lead_store_error_not_found_returns_404(self):
        exc = LeadStoreError("Lead not found", category="not_found")
        with patch("routers.leads.store_lead", side_effect=exc):
            resp = self.client.post("/leads/store", json=self.payload)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("Lead not found", resp.json().get("detail", ""))

    def test_2_unexpected_runtime_error_returns_500_generic_detail(self):
        with patch("routers.leads.store_lead", side_effect=RuntimeError("boom")):
            resp = self.client.post("/leads/store", json=self.payload)
        self.assertEqual(resp.status_code, 500)
        detail = resp.json().get("detail", "")
        self.assertNotIn("boom", detail)
        self.assertEqual(detail, "Unexpected error while storing the lead.")


class TestLeadsRouterGet(unittest.TestCase):
    """Tests for GET /leads/{lead_id}."""

    def setUp(self):
        self.client = _build_client()
        self.sample = _build_sample_record()

    def test_3_get_valid_uuid_returns_200(self):
        with patch("routers.leads.get_lead", return_value=self.sample):
            resp = self.client.get(f"/leads/{self.sample.id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["id"], self.sample.id)

    def test_3_get_invalid_uuid_returns_422(self):
        resp = self.client.get("/leads/abc")
        self.assertEqual(resp.status_code, 422)

    def test_3_get_test_endpoint_returns_original_test_message(self):
        resp = self.client.get("/leads/test")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"message": "Leads router is working correctly"})


class TestLeadsRouterFlags(unittest.TestCase):
    """Tests for PATCH /leads/{lead_id}/flags."""

    def setUp(self):
        self.client = _build_client()
        self.sample = _build_sample_record(email_sent=True)
        self.lead_id = self.sample.id

    def test_4_patch_flags_true_returns_200(self):
        with patch("routers.leads.update_flags", return_value=self.sample):
            resp = self.client.patch(
                f"/leads/{self.lead_id}/flags",
                json={"email_sent": True},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["email_sent"])

    def test_4_patch_flags_empty_payload_returns_422(self):
        resp = self.client.patch(f"/leads/{self.lead_id}/flags", json={})
        self.assertEqual(resp.status_code, 422)

    def test_4_patch_flags_false_flag_returns_422(self):
        resp = self.client.patch(
            f"/leads/{self.lead_id}/flags",
            json={"email_sent": False},
        )
        self.assertEqual(resp.status_code, 422)

    def test_4_patch_flags_not_found_returns_404(self):
        exc = LeadStoreError(f"Lead {self.lead_id} was not found.", category="not_found")
        with patch("routers.leads.update_flags", side_effect=exc):
            resp = self.client.patch(
                f"/leads/{self.lead_id}/flags",
                json={"email_sent": True},
            )
        self.assertEqual(resp.status_code, 404)
        self.assertIn("was not found", resp.json().get("detail", ""))


class TestLeadsRouterUnpatchedEnv(unittest.TestCase):
    """Tests with unpatched service functions and empty environment."""

    def test_5_unpatched_env_cleared_returns_500_mentioning_supabase_url(self):
        client = _build_client()
        sample = _build_sample_record()
        payload = sample.model_dump(mode="json")
        with patch.dict(os.environ, {}, clear=True):
            resp = client.post("/leads/store", json=payload)
        self.assertEqual(resp.status_code, 500)
        detail = resp.json().get("detail", "")
        self.assertIn("SUPABASE_URL", detail)


class TestLeadsRouterOpenAPI(unittest.TestCase):
    """Tests OpenAPI schema paths."""

    def test_6_openapi_schema_contains_leads_paths(self):
        app = FastAPI()
        app.include_router(leads_router.router)
        paths = app.openapi().get("paths", {})
        self.assertIn("/leads/store", paths)
        self.assertIn("/leads/{lead_id}", paths)
        self.assertIn("/leads/{lead_id}/flags", paths)


if __name__ == "__main__":
    unittest.main()
