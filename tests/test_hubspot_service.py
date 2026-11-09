"""
tests/test_hubspot_service.py

Unit and integration tests for services/hubspot_service.py.

Rules enforced:
- Only unittest, unittest.mock, httpx.MockTransport.
- No network access, no .env file read, no real HubSpot account touched.
- main.py is never imported.
- No importlib.reload calls.
- Pure ASCII and saved as UTF-8 without BOM.
- Env-dependent tests use patch.dict(os.environ, {...}, clear=True).
- Table-driven tests use subTest.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

from models import HubSpotContact, LeadInput, LeadRecord, LeadScore
from services.hubspot_service import (
    _BASE_URL,
    _API_VERSION,
    _CUSTOM_PROPERTIES,
    _STANDARD_PROPERTIES,
    CRMError,
    _build_payload,
    _build_properties,
    _classify_failure,
    _get_http_client,
    _is_dry_run,
    _parse_success,
    _split_name,
    contact_from_record,
    upsert_contact,
)


class TestHubSpotService(unittest.TestCase):
    """Test suite for the HubSpot service integration layer."""

    # -----------------------------------------------------------------------
    # A. Import safety
    # -----------------------------------------------------------------------
    def test_import_safety_fresh_copy_and_ascii(self) -> None:
        """Loading a fresh copy of hubspot_service with an empty environment

        must not raise, and the source file must decode as pure ASCII.
        """
        service_path = (
            Path(__file__).resolve().parent.parent
            / "services"
            / "hubspot_service.py"
        )
        source_bytes = service_path.read_bytes()

        # Assert source file decodes as pure ASCII
        try:
            source_text = source_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"services/hubspot_service.py contains non-ASCII bytes: {exc}")
        self.assertIsInstance(source_text, str)

        # Load fresh private copy under empty environment
        with patch.dict(os.environ, {}, clear=True):
            spec = importlib.util.spec_from_file_location(
                "hubspot_service_fresh", str(service_path)
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            fresh_module = importlib.util.module_from_spec(spec)
            # Must not raise
            spec.loader.exec_module(fresh_module)
            self.assertTrue(hasattr(fresh_module, "upsert_contact"))
            self.assertTrue(hasattr(fresh_module, "CRMError"))

    # -----------------------------------------------------------------------
    # B. Contract
    # -----------------------------------------------------------------------
    def test_contract_hubspot_contact_fields_and_custom_properties(self) -> None:
        """HubSpotContact fields must partition into email, standard, and custom;

        every _CUSTOM_PROPERTIES key and value must be non-empty.
        """
        expected_fields = {"email"} | set(_STANDARD_PROPERTIES) | set(_CUSTOM_PROPERTIES)
        actual_fields = set(HubSpotContact.model_fields.keys())
        self.assertEqual(
            actual_fields,
            expected_fields,
            "HubSpotContact model fields changed without a mapping update.",
        )

        for key, value in _CUSTOM_PROPERTIES.items():
            with self.subTest(key=key, value=value):
                self.assertTrue(isinstance(key, str) and len(key.strip()) > 0)
                self.assertTrue(isinstance(value, str) and len(value.strip()) > 0)

    # -----------------------------------------------------------------------
    # C. _split_name
    # -----------------------------------------------------------------------
    def test_split_name(self) -> None:
        """_split_name splits full names correctly, defaulting missing lastname to '-'."""
        cases = [
            ("Sarah Mitchell", ("Sarah", "Mitchell")),
            ("Maria de la Cruz", ("Maria", "de la Cruz")),
            ("Madonna", ("Madonna", "-")),
            ("  Jean   Luc  Picard ", ("Jean", "Luc Picard")),
        ]
        for full_name, expected in cases:
            with self.subTest(full_name=full_name):
                self.assertEqual(_split_name(full_name), expected)

    # -----------------------------------------------------------------------
    # D. contact_from_record
    # -----------------------------------------------------------------------
    def test_contact_from_record(self) -> None:
        """contact_from_record builds a valid HubSpotContact from a LeadRecord."""
        lead = LeadInput(
            name="Sarah Mitchell",
            email="sarah.mitchell@example.com",
            phone="+14155550192",
            message="Looking for a house in Austin",
            property_type=None,
            preferred_location=None,
        )
        score = LeadScore(
            score=87,
            reasoning="Qualified buyer with confirmed timeline",
            suggested_action="Schedule property viewing",
            priority_level="urgent",
        )
        record = LeadRecord(lead=lead, score=score)

        contact = contact_from_record(record)
        self.assertIsInstance(contact, HubSpotContact)
        self.assertEqual(contact.email, "sarah.mitchell@example.com")
        self.assertEqual(contact.firstname, "Sarah")
        self.assertEqual(contact.lastname, "Mitchell")
        self.assertEqual(contact.phone, "+14155550192")
        self.assertIsNone(contact.property_type)
        self.assertIsNone(contact.preferred_location)
        self.assertEqual(contact.lead_score, 87)
        self.assertEqual(contact.lead_status, record.score.category)
        self.assertEqual(contact.lead_status, "hot")

        # Test with optional property_type and preferred_location populated
        lead_with_opts = LeadInput(
            name="Maria de la Cruz",
            email="maria@example.com",
            phone="+14155550199",
            message="Looking for a condo",
            property_type="condo",
            preferred_location="Downtown",
        )
        score_warm = LeadScore(
            score=55,
            reasoning="Moderate intent",
            suggested_action="Follow up via email",
            priority_level="normal",
        )
        record_with_opts = LeadRecord(lead=lead_with_opts, score=score_warm)
        contact_opts = contact_from_record(record_with_opts)
        self.assertEqual(contact_opts.firstname, "Maria")
        self.assertEqual(contact_opts.lastname, "de la Cruz")
        self.assertEqual(contact_opts.property_type, "condo")
        self.assertEqual(contact_opts.preferred_location, "Downtown")
        self.assertEqual(contact_opts.lead_status, "warm")

    # -----------------------------------------------------------------------
    # E. _build_properties
    # -----------------------------------------------------------------------
    def test_build_properties(self) -> None:
        """_build_properties partitions standard/custom, strips phone, omits blanks,

        converts lead_score to str, and never includes email or empty strings.
        """
        contact = HubSpotContact(
            email="sarah.mitchell@example.com",
            firstname="Sarah",
            lastname="Mitchell",
            phone="  +14155550192  ",
            property_type=None,
            preferred_location="   ",
            lead_score=87,
            lead_status="hot",
        )
        properties, summary = _build_properties(contact)

        self.assertEqual(summary["standard"], ["firstname", "lastname", "phone"])
        self.assertEqual(summary["custom"], ["lead_score", "lead_status"])

        self.assertNotIn("email", properties)
        self.assertEqual(properties["phone"], "+14155550192")
        self.assertEqual(properties["firstname"], "Sarah")
        self.assertEqual(properties["lastname"], "Mitchell")
        self.assertEqual(properties["lead_score"], "87")
        self.assertIsInstance(properties["lead_score"], str)
        self.assertEqual(properties["lead_status"], "hot")

        # None and blank optional fields omitted
        self.assertNotIn("property_type", properties)
        self.assertNotIn("preferred_location", properties)

        # No empty string anywhere
        for k, v in properties.items():
            self.assertNotEqual(v, "", f"Property {k} is an empty string.")

    # -----------------------------------------------------------------------
    # F. _build_payload
    # -----------------------------------------------------------------------
    def test_build_payload(self) -> None:
        """_build_payload lower-cases email and sets idProperty='email'."""
        contact = HubSpotContact(
            email="Sarah.Mitchell@EXAMPLE.COM",
            firstname="Sarah",
            lastname="Mitchell",
            phone="+14155550192",
            lead_score=87,
            lead_status="hot",
        )
        payload, summary = _build_payload(contact)

        self.assertIn("inputs", payload)
        self.assertEqual(len(payload["inputs"]), 1)
        input_item = payload["inputs"][0]
        self.assertEqual(input_item["idProperty"], "email")
        self.assertEqual(input_item["id"], "sarah.mitchell@example.com")
        self.assertIn("properties", input_item)
        self.assertNotIn("email", input_item["properties"])
        self.assertEqual(summary["standard"], ["firstname", "lastname", "phone"])

    # -----------------------------------------------------------------------
    # G. _is_dry_run
    # -----------------------------------------------------------------------
    def test_is_dry_run(self) -> None:
        """_is_dry_run returns False ONLY for false, 0, no, off; True for anything else."""
        true_values = [None, "", "true", "yes", "garbage", "1", "TRUE", "True", "y"]
        for val in true_values:
            with self.subTest(val=val):
                env = {} if val is None else {"HUBSPOT_DRY_RUN": val}
                with patch.dict(os.environ, env, clear=True):
                    self.assertTrue(_is_dry_run())

        false_values = ["false", " FALSE ", "0", "no", "off", "False", "NO", "OFF"]
        for val in false_values:
            with self.subTest(val=val):
                with patch.dict(os.environ, {"HUBSPOT_DRY_RUN": val}, clear=True):
                    self.assertFalse(_is_dry_run())

    # -----------------------------------------------------------------------
    # H. Dry-run
    # -----------------------------------------------------------------------
    def test_upsert_contact_dry_run(self) -> None:
        """Dry-run returns expected response without checking token or making HTTP calls."""
        contact = HubSpotContact(
            email="sarah.mitchell@example.com",
            firstname="Sarah",
            lastname="Mitchell",
            phone="+14155550192",
            lead_score=87,
            lead_status="hot",
        )
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "services.hubspot_service._get_http_client",
                side_effect=AssertionError("HTTP client must not be accessed in dry-run mode"),
            ):
                result = upsert_contact(contact)

        self.assertTrue(result["dry_run"])
        self.assertIsNone(result["contact_id"])
        self.assertIsNone(result["created"])
        self.assertEqual(result["status"], "dry_run")
        self.assertIn("DRY RUN", result["message"])
        self.assertIn("standard", result["properties_sent"])
        self.assertIn("custom", result["properties_sent"])

    # -----------------------------------------------------------------------
    # I. Live success
    # -----------------------------------------------------------------------
    def test_upsert_contact_live_success_created_and_updated(self) -> None:
        """Live success parses contact_id, created flag, verifies request method/path/auth/body."""
        contact = HubSpotContact(
            email="sarah.mitchell@example.com",
            firstname="Sarah",
            lastname="Mitchell",
            phone="+14155550192",
            lead_score=87,
            lead_status="hot",
        )
        expected_payload, _ = _build_payload(contact)

        # 1. new: true -> created
        requests_created: list[httpx.Request] = []

        def handler_created(request: httpx.Request) -> httpx.Response:
            requests_created.append(request)
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETE",
                    "numErrors": 0,
                    "results": [{"id": "512", "new": True}],
                },
            )

        with patch.dict(
            os.environ,
            {"HUBSPOT_DRY_RUN": "false", "HUBSPOT_ACCESS_TOKEN": "test-token"},
            clear=True,
        ):
            with patch(
                "services.hubspot_service._get_http_client",
                return_value=httpx.Client(
                    transport=httpx.MockTransport(handler_created),
                    base_url="https://api.hubapi.com",
                    headers={"Authorization": "Bearer test-token"},
                ),
            ):
                result = upsert_contact(contact)

        self.assertEqual(result["contact_id"], "512")
        self.assertIs(result["created"], True)
        self.assertEqual(result["status"], "created")
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["message"], "HubSpot contact created")

        self.assertEqual(len(requests_created), 1)
        req = requests_created[0]
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.url.path, "/crm/objects/2026-09/contacts/batch/upsert")
        self.assertEqual(req.headers.get("Authorization"), "Bearer test-token")
        self.assertEqual(json.loads(req.content.decode("utf-8")), expected_payload)

        # 2. new: false -> updated
        requests_updated: list[httpx.Request] = []

        def handler_updated(request: httpx.Request) -> httpx.Response:
            requests_updated.append(request)
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETE",
                    "numErrors": 0,
                    "results": [{"id": "512", "new": False}],
                },
            )

        with patch.dict(
            os.environ,
            {"HUBSPOT_DRY_RUN": "false", "HUBSPOT_ACCESS_TOKEN": "test-token"},
            clear=True,
        ):
            with patch(
                "services.hubspot_service._get_http_client",
                return_value=httpx.Client(
                    transport=httpx.MockTransport(handler_updated),
                    base_url="https://api.hubapi.com",
                    headers={"Authorization": "Bearer test-token"},
                ),
            ):
                result = upsert_contact(contact)

        self.assertEqual(result["contact_id"], "512")
        self.assertIs(result["created"], False)
        self.assertEqual(result["status"], "updated")
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["message"], "HubSpot contact updated")
        self.assertEqual(len(requests_updated), 1)

    # -----------------------------------------------------------------------
    # J. 409 Conflict Retry
    # -----------------------------------------------------------------------
    def test_upsert_contact_409_retry_and_double_409(self) -> None:
        """A 409 triggers exactly one retry; two consecutive 409s raise CRMError conflict."""
        contact = HubSpotContact(
            email="sarah.mitchell@example.com",
            firstname="Sarah",
            lastname="Mitchell",
            phone="+14155550192",
            lead_score=87,
            lead_status="hot",
        )

        # 1. First 409, then 200 -> succeeds with 2 identical requests
        requests_retry: list[httpx.Request] = []
        sequence_retry = [
            httpx.Response(409, json={"status": "error", "message": "Conflict"}),
            httpx.Response(
                200,
                json={
                    "status": "COMPLETE",
                    "numErrors": 0,
                    "results": [{"id": "512", "new": False}],
                },
            ),
        ]

        def handler_409_success(request: httpx.Request) -> httpx.Response:
            requests_retry.append(request)
            return sequence_retry.pop(0)

        with patch.dict(
            os.environ,
            {"HUBSPOT_DRY_RUN": "false", "HUBSPOT_ACCESS_TOKEN": "test-token"},
            clear=True,
        ):
            with patch(
                "services.hubspot_service._get_http_client",
                return_value=httpx.Client(
                    transport=httpx.MockTransport(handler_409_success),
                    base_url="https://api.hubapi.com",
                    headers={"Authorization": "Bearer test-token"},
                ),
            ):
                res = upsert_contact(contact)

        self.assertEqual(res["contact_id"], "512")
        self.assertEqual(len(requests_retry), 2)
        self.assertEqual(requests_retry[0].content, requests_retry[1].content)

        # 2. Two consecutive 409s -> CRMError conflict
        requests_double_409: list[httpx.Request] = []

        def handler_double_409(request: httpx.Request) -> httpx.Response:
            requests_double_409.append(request)
            return httpx.Response(
                409, json={"status": "error", "message": "Conflict", "correlationId": "cid-409"}
            )

        with patch.dict(
            os.environ,
            {"HUBSPOT_DRY_RUN": "false", "HUBSPOT_ACCESS_TOKEN": "test-token"},
            clear=True,
        ):
            with patch(
                "services.hubspot_service._get_http_client",
                return_value=httpx.Client(
                    transport=httpx.MockTransport(handler_double_409),
                    base_url="https://api.hubapi.com",
                    headers={"Authorization": "Bearer test-token"},
                ),
            ):
                with self.assertRaises(CRMError) as ctx:
                    upsert_contact(contact)

        self.assertEqual(ctx.exception.category, "conflict")
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(len(requests_double_409), 2)

    # -----------------------------------------------------------------------
    # K & L. Failure Classification and Safety
    # -----------------------------------------------------------------------
    def test_failure_classification_and_safety(self) -> None:
        """Test failure classification mapping and PII/canary safety across all failure modes."""
        contact = HubSpotContact(
            email="sarah.mitchell@example.com",
            firstname="Sarah",
            lastname="Mitchell",
            phone="+14155550192",
            lead_score=87,
            lead_status="hot",
        )
        token = "test-token-secret-xyz"
        canary = "CANARY-123 sarah@example.com"
        cid = "corr-test-id-999"

        cases = [
            (
                "401_config",
                lambda req: httpx.Response(
                    401, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "config",
                401,
                cid,
                lambda err_str: None,
            ),
            (
                "403_config_scopes",
                lambda req: httpx.Response(
                    403, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "config",
                403,
                cid,
                lambda err_str: (
                    self.assertIn("crm.objects.contacts.read", err_str),
                    self.assertIn("crm.objects.contacts.write", err_str),
                ),
            ),
            (
                "404_config_api_version",
                lambda req: httpx.Response(
                    404, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "config",
                404,
                cid,
                lambda err_str: self.assertIn("_API_VERSION", err_str),
            ),
            (
                "429_transient",
                lambda req: httpx.Response(
                    429, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "transient",
                429,
                cid,
                lambda err_str: None,
            ),
            (
                "500_transient",
                lambda req: httpx.Response(
                    500, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "transient",
                500,
                cid,
                lambda err_str: None,
            ),
            (
                "502_transient",
                lambda req: httpx.Response(
                    502, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "transient",
                502,
                cid,
                lambda err_str: None,
            ),
            (
                "503_transient",
                lambda req: httpx.Response(
                    503, json={"status": "error", "message": canary, "correlationId": cid}
                ),
                "transient",
                503,
                cid,
                lambda err_str: None,
            ),
            (
                "non_json_502_transient",
                lambda req: httpx.Response(502, text=f"<html>{canary} Bad Gateway</html>"),
                "transient",
                502,
                None,
                lambda err_str: None,
            ),
            (
                "400_property_doesnt_exist",
                lambda req: httpx.Response(
                    400,
                    json={
                        "status": "error",
                        "message": f"{canary} PROPERTY_DOESNT_EXIST: lead_score",
                        "correlationId": cid,
                    },
                ),
                "config",
                400,
                cid,
                lambda err_str: (
                    self.assertIn("lead_score", err_str),
                    self.assertIn("lead_status", err_str),
                    self.assertIn("property_type", err_str),
                    self.assertIn("preferred_location", err_str),
                ),
            ),
            (
                "400_invalid_email",
                lambda req: httpx.Response(
                    400,
                    json={
                        "status": "error",
                        "message": f"{canary} INVALID_EMAIL",
                        "correlationId": cid,
                    },
                ),
                "invalid",
                400,
                cid,
                lambda err_str: None,
            ),
            (
                "400_invalid_option",
                lambda req: httpx.Response(
                    400,
                    json={
                        "status": "error",
                        "message": f"{canary} INVALID_OPTION",
                        "correlationId": cid,
                    },
                ),
                "config",
                400,
                cid,
                lambda err_str: None,
            ),
            (
                "400_other",
                lambda req: httpx.Response(
                    400,
                    json={
                        "status": "error",
                        "category": "VALIDATION_ERROR",
                        "message": canary,
                        "correlationId": cid,
                    },
                ),
                "config",
                400,
                cid,
                lambda err_str: None,
            ),
            (
                "200_numErrors_1_with_errors_list",
                lambda req: httpx.Response(
                    200,
                    json={
                        "status": "COMPLETE",
                        "numErrors": 1,
                        "errors": [
                            {
                                "status": 400,
                                "message": f"{canary} PROPERTY_DOESNT_EXIST",
                            }
                        ],
                        "results": [],
                        "correlationId": cid,
                    },
                ),
                "config",
                400,
                cid,
                lambda err_str: (
                    self.assertIn("lead_score", err_str),
                    self.assertIn("lead_status", err_str),
                ),
            ),
            (
                "200_empty_results",
                lambda req: httpx.Response(
                    200,
                    json={"status": "COMPLETE", "numErrors": 0, "results": []},
                ),
                "config",
                200,
                None,
                lambda err_str: self.assertIn("unexpected response", err_str),
            ),
            (
                "connect_timeout",
                lambda req: (_ for _ in ()).throw(httpx.ConnectTimeout("t", request=req)),
                "transient",
                None,
                None,
                lambda err_str: None,
            ),
            (
                "200_numErrors_1_no_errors_list",
                lambda req: httpx.Response(
                    200,
                    json={
                        "status": "COMPLETE",
                        "numErrors": 1,
                        "results": [],
                        "correlationId": cid,
                    },
                ),
                "config",
                400,
                cid,
                lambda err_str: (
                    self.assertIn("HTTP 400", err_str),
                    self.assertNotIn("HTTP 200", err_str),
                ),
            ),
        ]

        for name, handler_fn, exp_cat, exp_status, exp_cid, extra_assert in cases:
            with self.subTest(case=name):
                with patch.dict(
                    os.environ,
                    {"HUBSPOT_DRY_RUN": "false", "HUBSPOT_ACCESS_TOKEN": token},
                    clear=True,
                ):
                    with patch(
                        "services.hubspot_service._get_http_client",
                        return_value=httpx.Client(
                            transport=httpx.MockTransport(handler_fn),
                            base_url="https://api.hubapi.com",
                            headers={"Authorization": f"Bearer {token}"},
                        ),
                    ):
                        with self.assertRaises(CRMError) as ctx:
                            upsert_contact(contact)

                exc = ctx.exception
                self.assertEqual(exc.category, exp_cat)
                if exp_status is not None:
                    self.assertEqual(exc.status, exp_status)

                err_text = str(exc)

                # Safety checks: contact PII, token, and canary text never appear in error
                self.assertNotIn("sarah.mitchell@example.com", err_text)
                self.assertNotIn("Sarah", err_text)
                self.assertNotIn("Mitchell", err_text)
                self.assertNotIn("+14155550192", err_text)
                self.assertNotIn(token, err_text)
                self.assertNotIn("CANARY-123", err_text)

                if exp_cid is not None:
                    self.assertEqual(exc.correlation_id, exp_cid)
                    self.assertIn(exp_cid, err_text)
                else:
                    self.assertIsNone(exc.correlation_id)

                extra_assert(err_text)

    # -----------------------------------------------------------------------
    # M. Environment
    # -----------------------------------------------------------------------
    def test_environment_missing_token_and_legacy_key(self) -> None:
        """Missing token in live mode raises config error; legacy HUBSPOT_API_KEY is ignored."""
        contact = HubSpotContact(
            email="sarah.mitchell@example.com",
            firstname="Sarah",
            lastname="Mitchell",
            phone="+14155550192",
            lead_score=87,
            lead_status="hot",
        )

        # 1. Missing HUBSPOT_ACCESS_TOKEN
        with patch.dict(os.environ, {"HUBSPOT_DRY_RUN": "false"}, clear=True):
            with self.assertRaises(CRMError) as ctx:
                upsert_contact(contact)
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("HUBSPOT_ACCESS_TOKEN", str(ctx.exception))

        # 2. Only legacy HUBSPOT_API_KEY present
        with patch.dict(
            os.environ,
            {"HUBSPOT_DRY_RUN": "false", "HUBSPOT_API_KEY": "legacy-key-123"},
            clear=True,
        ):
            with self.assertRaises(CRMError) as ctx:
                upsert_contact(contact)
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("HUBSPOT_ACCESS_TOKEN", str(ctx.exception))

    # -----------------------------------------------------------------------
    # N. _get_http_client
    # -----------------------------------------------------------------------
    def test_get_http_client_properties(self) -> None:
        """_get_http_client configures base_url, timeout, and Authorization header."""
        test_token = "my-secret-access-token"
        with patch.dict(os.environ, {"HUBSPOT_ACCESS_TOKEN": test_token}, clear=True):
            client = _get_http_client()
            try:
                self.assertEqual(str(client.base_url).rstrip("/"), "https://api.hubapi.com")
                self.assertEqual(client.headers.get("Authorization"), f"Bearer {test_token}")
                self.assertEqual(client.headers.get("Accept"), "application/json")
                self.assertEqual(client.timeout, httpx.Timeout(10))
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
