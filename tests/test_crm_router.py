"""
tests/test_crm_router.py

Integration tests for the /crm/contact HTTP endpoint (routers/crm.py).

Rules enforced:
- Only unittest, unittest.mock, and fastapi.testclient.TestClient.
- No network access, no .env file read.
- main.py is never imported.
- No importlib.reload calls.
- Pure ASCII and saved as UTF-8 without BOM.
- Router tests build a throwaway FastAPI app: app = FastAPI(); app.include_router(crm.router).
- Table-driven tests use subTest.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.crm as crm
from services.hubspot_service import CRMError


def _build_test_client() -> TestClient:
    """Build a throwaway FastAPI test client with only the CRM router included."""
    app = FastAPI()
    app.include_router(crm.router)
    return TestClient(app, raise_server_exceptions=False)


class TestCRMRouter(unittest.TestCase):
    """Test suite for the CRM router HTTP endpoints."""

    def setUp(self) -> None:
        self.client = _build_test_client()
        self.valid_payload = {
            "email": "sarah.mitchell@example.com",
            "firstname": "Sarah",
            "lastname": "Mitchell",
            "phone": "+14155550192",
            "property_type": "house",
            "preferred_location": "Austin, TX",
            "lead_score": 87,
            "lead_status": "hot",
        }

    # -----------------------------------------------------------------------
    # 1. Real dry-run path (env cleared, upsert_contact NOT patched)
    # -----------------------------------------------------------------------
    def test_real_dry_run_path(self) -> None:
        """POST /crm/contact with cleared env executes unpatched dry-run path returning 200."""
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post("/crm/contact", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["dry_run"])
        self.assertIsNone(data["contact_id"])
        self.assertIsNone(data["created"])
        self.assertEqual(data["status"], "dry_run")
        self.assertIn("DRY RUN", data["message"])

        props_sent = data["properties_sent"]
        self.assertEqual(props_sent["standard"], ["firstname", "lastname", "phone"])
        self.assertEqual(
            props_sent["custom"],
            ["lead_score", "lead_status", "property_type", "preferred_location"],
        )

    # -----------------------------------------------------------------------
    # 2. Patched upsert_contact returning live-shaped result
    # -----------------------------------------------------------------------
    def test_live_upsert_success(self) -> None:
        """Patched upsert_contact returning live-mode success returns 200 with contact_id and created."""
        live_result = {
            "message": "HubSpot contact created",
            "contact_id": "512",
            "created": True,
            "status": "created",
            "dry_run": False,
            "properties_sent": {
                "standard": ["firstname", "lastname", "phone"],
                "custom": [
                    "lead_score",
                    "lead_status",
                    "property_type",
                    "preferred_location",
                ],
            },
        }

        with patch("routers.crm.upsert_contact", return_value=live_result):
            response = self.client.post("/crm/contact", json=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["dry_run"])
        self.assertEqual(data["contact_id"], "512")
        self.assertIs(data["created"], True)
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["message"], "HubSpot contact created")

    # -----------------------------------------------------------------------
    # 3. Error mapping
    # -----------------------------------------------------------------------
    def test_error_mapping(self) -> None:
        """CRMError categories, ValueError, and generic exceptions map to expected status codes."""
        error_cases = [
            (
                "invalid_category_422",
                CRMError("HubSpot rejected invalid contact data.", category="invalid"),
                422,
                "HubSpot rejected invalid contact data.",
            ),
            (
                "conflict_category_409",
                CRMError("Duplicate conflicting contact exists.", category="conflict"),
                409,
                "Duplicate conflicting contact exists.",
            ),
            (
                "config_category_500",
                CRMError("Custom property missing in HubSpot.", category="config"),
                500,
                "Custom property missing in HubSpot.",
            ),
            (
                "transient_category_502",
                CRMError("HubSpot unavailable.", category="transient"),
                502,
                "HubSpot unavailable.",
            ),
            (
                "value_error_422",
                ValueError("Invalid value provided."),
                422,
                "Invalid value provided.",
            ),
            (
                "generic_runtime_error_500_masked",
                RuntimeError("boom"),
                500,
                "Unexpected error while syncing the contact.",
            ),
        ]

        for name, exc, expected_status, expected_detail in error_cases:
            with self.subTest(case=name):
                with patch("routers.crm.upsert_contact", side_effect=exc):
                    response = self.client.post("/crm/contact", json=self.valid_payload)

                self.assertEqual(response.status_code, expected_status)
                detail = response.json().get("detail")
                self.assertEqual(detail, expected_detail)
                # When runtime error is masked, "boom" must never appear in response
                if isinstance(exc, RuntimeError) and not isinstance(exc, CRMError):
                    self.assertNotIn("boom", detail)

    # -----------------------------------------------------------------------
    # 4. Payload validation
    # -----------------------------------------------------------------------
    def test_payload_validation_rejects_invalid_inputs(self) -> None:
        """Pydantic validation rejects empty firstname, invalid lead_status, and score > 100 with 422."""
        invalid_cases = [
            (
                "empty_firstname",
                {**self.valid_payload, "firstname": "   "},
            ),
            (
                "invalid_lead_status_lukewarm",
                {**self.valid_payload, "lead_status": "lukewarm"},
            ),
            (
                "score_out_of_bounds_150",
                {**self.valid_payload, "lead_score": 150},
            ),
        ]

        for name, invalid_body in invalid_cases:
            with self.subTest(case=name):
                response = self.client.post("/crm/contact", json=invalid_body)
                self.assertEqual(response.status_code, 422)

    # -----------------------------------------------------------------------
    # 5. OpenAPI schema
    # -----------------------------------------------------------------------
    def test_openapi_schema_contains_crm_contact(self) -> None:
        """The OpenAPI schema documents /crm/contact endpoint and its responses."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        paths = schema.get("paths", {})
        self.assertIn("/crm/contact", paths)
        self.assertIn("post", paths["/crm/contact"])
        responses = paths["/crm/contact"]["post"]["responses"]
        for expected_code in ("200", "409", "422", "500", "502"):
            self.assertIn(
                expected_code,
                responses,
                f"Status code {expected_code} missing from OpenAPI schema.",
            )

    # -----------------------------------------------------------------------
    # 6. main.py registration (static check)
    # -----------------------------------------------------------------------
    def test_main_py_registration_static_check(self) -> None:
        """main.py statically imports crm router and calls app.include_router(crm.router)."""
        main_py_path = Path(__file__).resolve().parent.parent / "main.py"
        content = main_py_path.read_text(encoding="utf-8")

        router_import_lines = [
            line for line in content.splitlines() if line.strip().startswith("from routers import")
        ]
        self.assertTrue(
            len(router_import_lines) > 0,
            "No 'from routers import ...' line found in main.py.",
        )
        import_line = router_import_lines[0]
        imported_names = [
            name.strip() for name in import_line.replace("from routers import", "").split(",")
        ]
        self.assertIn("crm", imported_names, f"'crm' not in import line: {import_line}")
        self.assertIn(
            "app.include_router(crm.router)",
            content,
            "app.include_router(crm.router) not found in main.py.",
        )


if __name__ == "__main__":
    unittest.main()
