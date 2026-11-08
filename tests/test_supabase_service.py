"""
tests/test_supabase_service.py

Unit tests for services/supabase_service.py.

Rules enforced:
- No network access; .env is never read.
- Env-dependent tests use patch.dict(os.environ, {...}, clear=True).
- main.py is never imported.
- All database operations are mocked via FakeClient.
"""

import base64
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from postgrest.exceptions import APIError

from models import LeadInput, LeadRecord, LeadScore
import services.supabase_service as svc


def _make_api_error(
    message: str = "Database error",
    code: str | int | None = None,
    hint: str | None = None,
    details: str | None = None,
) -> APIError:
    """Build an APIError instance matching the installed class signature."""
    return APIError({
        "message": message,
        "code": code,
        "hint": hint,
        "details": details,
    })


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


# =========================================================================== #
# FAKE CLIENT & QUERY BUILDER                                                 #
# =========================================================================== #

class FakeQueryBuilder:
    """Chainable PostgREST query builder simulation for FakeClient."""

    def __init__(self, client: "FakeClient"):
        self.client = client
        self.action = None
        self.row_to_insert = None
        self.update_values = None
        self.eq_filters = []
        self.limit_val = None

    def insert(self, row: dict):
        self.action = "insert"
        self.row_to_insert = dict(row)
        return self

    def select(self, cols: str = "*"):
        self.action = "select"
        self.cols = cols
        return self

    def update(self, values: dict):
        self.action = "update"
        self.update_values = dict(values)
        self.client.recorded_updates.append(dict(values))
        return self

    def eq(self, col: str, val: str):
        self.eq_filters.append((col, val))
        return self

    def limit(self, n: int):
        self.limit_val = n
        return self

    def execute(self):
        if self.client._next_exc is not None:
            exc = self.client._next_exc
            self.client._next_exc = None
            raise exc

        if self.action == "insert":
            lead_id = self.row_to_insert["id"]
            if lead_id in self.client.storage:
                raise _make_api_error(
                    message="duplicate key value violates unique constraint",
                    code="23505",
                )
            stored = dict(self.row_to_insert)
            self.client.storage[lead_id] = stored
            return SimpleNamespace(data=[dict(stored)])

        elif self.action == "select":
            results = []
            for r in self.client.storage.values():
                match = True
                for col, val in self.eq_filters:
                    if r.get(col) != val:
                        match = False
                        break
                if match:
                    results.append(dict(r))
            if self.limit_val is not None:
                results = results[: self.limit_val]
            return SimpleNamespace(data=results)

        elif self.action == "update":
            results = []
            for r in self.client.storage.values():
                match = True
                for col, val in self.eq_filters:
                    if r.get(col) != val:
                        match = False
                        break
                if match:
                    r.update(self.update_values)
                    results.append(dict(r))
            return SimpleNamespace(data=results)

        return SimpleNamespace(data=[])


class FakeClient:
    """In-memory Supabase client simulation keyed by lead id."""

    def __init__(self, initial_rows: dict[str, dict] | None = None):
        self.storage = dict(initial_rows) if initial_rows else {}
        self._next_exc = None
        self.recorded_updates = []

    def table(self, name: str):
        assert name == "leads", f"Expected table 'leads', got '{name}'"
        return FakeQueryBuilder(self)

    def fail_next(self, exc: Exception):
        self._next_exc = exc


# =========================================================================== #
# A. IMPORT SAFETY                                                            #
# =========================================================================== #

class TestImportSafety(unittest.TestCase):
    """Importing with an empty environment must never raise."""

    def test_import_with_empty_env_does_not_raise(self):
        try:
            with patch.dict(os.environ, {}, clear=True):
                import services.supabase_service as fresh_svc
                importlib.reload(fresh_svc)
        except Exception as exc:
            self.fail(f"Import raised {type(exc).__name__}: {exc}")

    def test_source_file_is_ascii(self):
        src_path = inspect.getfile(svc)
        with open(src_path, "rb") as f:
            raw = f.read()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"supabase_service.py is not pure ASCII: {exc}")


# =========================================================================== #
# B. _to_uuid                                                                 #
# =========================================================================== #

class TestToUuid(unittest.TestCase):
    """_to_uuid validates and normalises UUID strings."""

    def test_valid_uuid_forms_canonicalized(self):
        canonical = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        cases = [
            "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "3FA85F64-5717-4562-B3FC-2C963F66AFA6",
            "3fa85f6457174562b3fc2c963f66afa6",
            "3FA85F6457174562B3FC2C963F66AFA6",
        ]
        for val in cases:
            with self.subTest(val=val):
                self.assertEqual(svc._to_uuid(val), canonical)

    def test_invalid_uuid_raises_value_error(self):
        invalid_cases = ["abc", "", "   ", "not-a-valid-uuid", "12345"]
        for val in invalid_cases:
            with self.subTest(val=val):
                with self.assertRaises(ValueError):
                    svc._to_uuid(val)


# =========================================================================== #
# C. _record_to_row                                                           #
# =========================================================================== #

class TestRecordToRow(unittest.TestCase):
    """_record_to_row flattens a LeadRecord to the 18 expected database keys."""

    EXPECTED_18_KEYS = {
        "id", "created_at", "name", "email", "phone", "message",
        "property_type", "preferred_location", "budget", "timeline",
        "score", "category", "reasoning", "suggested_action", "priority_level",
        "email_sent", "sms_sent", "hubspot_created",
    }

    def test_exact_18_keys(self):
        record = _build_sample_record()
        row = svc._record_to_row(record)
        self.assertEqual(set(row.keys()), self.EXPECTED_18_KEYS)
        self.assertEqual(len(row), 18)

    def test_values_flattened_correctly(self):
        record = _build_sample_record()
        row = svc._record_to_row(record)
        self.assertEqual(row["id"], "3fa85f64-5717-4562-b3fc-2c963f66afa6")
        self.assertEqual(row["name"], "Sarah Mitchell")
        self.assertEqual(row["email"], "sarah.mitchell@example.com")
        self.assertEqual(row["phone"], "+1-415-555-0192")
        self.assertEqual(row["message"], "Looking for a 3-bedroom house in Austin.")
        self.assertEqual(row["property_type"], "house")
        self.assertEqual(row["preferred_location"], "Austin, TX")
        self.assertEqual(row["budget"], "$450k-$550k")
        self.assertEqual(row["timeline"], "within 3 months")
        self.assertEqual(row["score"], 87)
        self.assertEqual(row["category"], "hot")
        self.assertEqual(row["reasoning"], "Clear budget and timeline.")
        self.assertEqual(row["suggested_action"], "Contact today.")
        self.assertEqual(row["priority_level"], "urgent")
        self.assertFalse(row["email_sent"])
        self.assertFalse(row["sms_sent"])
        self.assertFalse(row["hubspot_created"])

    def test_created_at_is_iso8601_with_timezone(self):
        record = _build_sample_record(
            created_at=datetime(2026, 4, 25, 21, 0, 0, tzinfo=timezone.utc)
        )
        row = svc._record_to_row(record)
        self.assertIn("+00:00", row["created_at"])

    def test_naive_created_at_treated_as_utc(self):
        naive_dt = datetime(2026, 4, 25, 21, 0, 0)
        record = _build_sample_record(created_at=naive_dt)
        row = svc._record_to_row(record)
        self.assertTrue(
            row["created_at"].endswith("+00:00") or row["created_at"].endswith("Z")
        )


# =========================================================================== #
# D. _row_to_record                                                           #
# =========================================================================== #

class TestRowToRecord(unittest.TestCase):
    """_row_to_record reconstitutes LeadRecord and enforces schema safety."""

    def test_round_trip_with_record_to_row(self):
        record = _build_sample_record()
        row = svc._record_to_row(record)
        reconstituted = svc._row_to_record(row)
        self.assertEqual(record.model_dump(), reconstituted.model_dump())

    def test_missing_column_raises_config_error(self):
        record = _build_sample_record()
        row = svc._record_to_row(record)
        del row["email"]
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc._row_to_record(row)
        self.assertEqual(ctx.exception.category, "config")
        self.assertIn("missing column 'email'", str(ctx.exception))

    def test_row_with_score_500_raises_config_error_without_leaking_email(self):
        record = _build_sample_record(email="confidential_lead@secret.com")
        row = svc._record_to_row(record)
        row["score"] = 500
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc._row_to_record(row)
        self.assertEqual(ctx.exception.category, "config")
        self.assertNotIn("confidential_lead@secret.com", str(ctx.exception))


# =========================================================================== #
# E. _check_key_kind                                                          #
# =========================================================================== #

class TestCheckKeyKind(unittest.TestCase):
    """_check_key_kind rejects publishable and anon keys; allows secret keys."""

    def test_publishable_key_raises_config_error(self):
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc._check_key_kind("sb_publishable_1234567890")
        self.assertEqual(ctx.exception.category, "config")
        self.assertIn("publishable key", str(ctx.exception))

    def test_jwt_with_anon_role_raises_config_error(self):
        header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode("ascii").rstrip("=")
        payload = base64.urlsafe_b64encode(b'{"role":"anon"}').decode("ascii").rstrip("=")
        sig = "signature"
        anon_jwt = f"{header}.{payload}.{sig}"
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc._check_key_kind(anon_jwt)
        self.assertEqual(ctx.exception.category, "config")
        self.assertIn("anon key", str(ctx.exception))

    def test_service_role_jwt_passes(self):
        header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').decode("ascii").rstrip("=")
        payload = base64.urlsafe_b64encode(b'{"role":"service_role"}').decode("ascii").rstrip("=")
        sig = "signature"
        service_jwt = f"{header}.{payload}.{sig}"
        svc._check_key_kind(service_jwt)

    def test_sb_secret_key_passes(self):
        svc._check_key_kind("sb_secret_1234567890abcdef")

    def test_garbage_string_passes(self):
        svc._check_key_kind("random_unparseable_non_jwt_key")


# =========================================================================== #
# F. _get_client                                                              #
# =========================================================================== #

class TestGetClient(unittest.TestCase):
    """_get_client validates environment configuration and instantiates client."""

    def test_missing_supabase_url_raises_config_error(self):
        with patch.dict(os.environ, {"SUPABASE_KEY": "sb_secret_abc"}, clear=True):
            with self.assertRaises(svc.LeadStoreError) as ctx:
                svc._get_client()
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("SUPABASE_URL is missing or empty", str(ctx.exception))

    def test_missing_supabase_key_raises_config_error(self):
        with patch.dict(os.environ, {"SUPABASE_URL": "https://xyz.supabase.co"}, clear=True):
            with self.assertRaises(svc.LeadStoreError) as ctx:
                svc._get_client()
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("SUPABASE_KEY is missing or empty", str(ctx.exception))

    def test_http_url_raises_config_error(self):
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "http://x.supabase.co", "SUPABASE_KEY": "sb_secret_abc"},
            clear=True,
        ):
            with self.assertRaises(svc.LeadStoreError) as ctx:
                svc._get_client()
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("SUPABASE_URL must be the project root", str(ctx.exception))

    def test_url_containing_rest_v1_raises_config_error(self):
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co/rest/v1", "SUPABASE_KEY": "sb_secret_abc"},
            clear=True,
        ):
            with self.assertRaises(svc.LeadStoreError) as ctx:
                svc._get_client()
            self.assertEqual(ctx.exception.category, "config")
            self.assertIn("SUPABASE_URL must be the project root", str(ctx.exception))

    @patch("services.supabase_service.create_client")
    @patch("services.supabase_service.ClientOptions")
    def test_client_creation_parameters(self, mock_client_options, mock_create_client):
        mock_options_instance = MagicMock()
        mock_client_options.return_value = mock_options_instance
        mock_client_instance = MagicMock()
        mock_create_client.return_value = mock_client_instance

        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co/", "SUPABASE_KEY": "sb_secret_abc"},
            clear=True,
        ):
            client = svc._get_client()
            self.assertEqual(client, mock_client_instance)

            # Trailing slash must be stripped
            mock_create_client.assert_called_once_with(
                "https://x.supabase.co",
                "sb_secret_abc",
                options=mock_options_instance,
            )
            mock_client_options.assert_called_once_with(
                postgrest_client_timeout=10,
                auto_refresh_token=False,
                persist_session=False,
            )


# =========================================================================== #
# G. _classify_error                                                          #
# =========================================================================== #

class TestClassifyError(unittest.TestCase):
    """_classify_error categorizes exceptions into config, transient, or not_found."""

    def test_config_codes(self):
        codes = [
            "42501", "42P01", "PGRST205", "PGRST301", "23514",
            "23502", "22P02", "22001", "404",
        ]
        for code in codes:
            with self.subTest(code=code):
                exc = _make_api_error(code=code, details="secret_leak_details")
                err = svc._classify_error(exc)
                self.assertEqual(err.category, "config")
                self.assertEqual(err.db_code, code)
                self.assertNotIn("secret_leak_details", str(err))

    def test_transient_codes(self):
        # String codes
        string_codes = ["PGRST002", "57014", "503", "500", "599"]
        for code in string_codes:
            with self.subTest(code=code):
                exc = _make_api_error(code=code, details="transient_details")
                err = svc._classify_error(exc)
                self.assertEqual(err.category, "transient")
                self.assertEqual(err.db_code, code)
                self.assertNotIn("transient_details", str(err))

        # Integer codes 502, 504
        for int_code in [502, 504]:
            with self.subTest(code=int_code):
                exc = _make_api_error(code=int_code, details="transient_details")
                err = svc._classify_error(exc)
                self.assertEqual(err.category, "transient")
                self.assertEqual(err.db_code, str(int_code))
                self.assertNotIn("transient_details", str(err))

    def test_httpx_connect_timeout_is_transient(self):
        exc = httpx.ConnectTimeout("Connection timed out")
        err = svc._classify_error(exc)
        self.assertEqual(err.category, "transient")
        self.assertIn("Could not reach Supabase", str(err))

    def test_value_error_is_config(self):
        exc = ValueError("Invalid input")
        err = svc._classify_error(exc)
        self.assertEqual(err.category, "config")
        self.assertIn("ValueError", str(err))

    def test_existing_lead_store_error_returned_unchanged(self):
        original = svc.LeadStoreError("Already wrapped", category="not_found", db_code="404")
        self.assertIs(svc._classify_error(original), original)


# =========================================================================== #
# H. store_lead SEMANTICS WITH FakeClient                                     #
# =========================================================================== #

class TestStoreLeadSemantics(unittest.TestCase):
    """store_lead ensures first-write-wins and OR-merged delivery flags."""

    def setUp(self):
        self.client = FakeClient()
        self.patcher = patch("services.supabase_service._get_client", return_value=self.client)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_new_id_created_true(self):
        rec = _build_sample_record()
        stored, created = svc.store_lead(rec)
        self.assertTrue(created)
        self.assertEqual(stored.id, rec.id)
        self.assertEqual(stored.lead.name, rec.lead.name)
        self.assertIn(rec.id, self.client.storage)

    def test_duplicate_id_preserves_original_name_and_score(self):
        rec1 = _build_sample_record(name="Original Name", score=85)
        stored1, created1 = svc.store_lead(rec1)
        self.assertTrue(created1)

        # Retry with different name, score, and reasoning
        rec2 = _build_sample_record(name="Updated Name", score=25, reasoning="Changed")
        stored2, created2 = svc.store_lead(rec2)
        self.assertFalse(created2)
        self.assertEqual(stored2.lead.name, "Original Name")
        self.assertEqual(stored2.score.score, 85)

    def test_flag_merge_only_sets_true_never_resets_to_false(self):
        # 1. Store with all flags False
        rec1 = _build_sample_record(email_sent=False, sms_sent=False, hubspot_created=False)
        stored1, _ = svc.store_lead(rec1)
        self.assertFalse(stored1.email_sent)

        # 2. Store again with email_sent=True -> row flag becomes True
        rec2 = _build_sample_record(email_sent=True, sms_sent=False, hubspot_created=False)
        stored2, created2 = svc.store_lead(rec2)
        self.assertFalse(created2)
        self.assertTrue(stored2.email_sent)

        # 3. Store again with email_sent=False -> row flag remains True
        rec3 = _build_sample_record(email_sent=False, sms_sent=False, hubspot_created=False)
        stored3, created3 = svc.store_lead(rec3)
        self.assertFalse(created3)
        self.assertTrue(stored3.email_sent)

    def test_invalid_id_raises_value_error(self):
        rec = _build_sample_record()
        object.__setattr__(rec, "id", "not-a-uuid")
        with self.assertRaises(ValueError):
            svc.store_lead(rec)

    def test_fail_next_transient_raises_lead_store_error_transient(self):
        rec = _build_sample_record()
        self.client.fail_next(httpx.ConnectTimeout("Network timeout"))
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc.store_lead(rec)
        self.assertEqual(ctx.exception.category, "transient")

    def test_fail_next_42501_raises_lead_store_error_config(self):
        rec = _build_sample_record()
        self.client.fail_next(_make_api_error(code="42501", message="Access denied"))
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc.store_lead(rec)
        self.assertEqual(ctx.exception.category, "config")


# =========================================================================== #
# I. get_lead                                                                 #
# =========================================================================== #

class TestGetLead(unittest.TestCase):
    """get_lead fetches an existing lead by UUID or raises appropriate error."""

    def setUp(self):
        self.rec = _build_sample_record()
        row = svc._record_to_row(self.rec)
        self.client = FakeClient({self.rec.id: row})
        self.patcher = patch("services.supabase_service._get_client", return_value=self.client)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_lead_found(self):
        fetched = svc.get_lead(self.rec.id)
        self.assertEqual(fetched.id, self.rec.id)
        self.assertEqual(fetched.lead.name, self.rec.lead.name)

    def test_unknown_valid_uuid_raises_not_found(self):
        unknown_id = "00000000-0000-0000-0000-000000000000"
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc.get_lead(unknown_id)
        self.assertEqual(ctx.exception.category, "not_found")
        self.assertIn("was not found", str(ctx.exception))

    def test_bad_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            svc.get_lead("not-a-valid-uuid")


# =========================================================================== #
# J. update_flags                                                             #
# =========================================================================== #

class TestUpdateFlags(unittest.TestCase):
    """update_flags sets only True flags and asserts False is never written."""

    def setUp(self):
        self.rec = _build_sample_record(email_sent=False, sms_sent=False, hubspot_created=False)
        row = svc._record_to_row(self.rec)
        self.client = FakeClient({self.rec.id: row})
        self.patcher = patch("services.supabase_service._get_client", return_value=self.client)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_sets_only_true_flags(self):
        updated = svc.update_flags(self.rec.id, email_sent=True)
        self.assertTrue(updated.email_sent)
        self.assertFalse(updated.sms_sent)
        self.assertFalse(updated.hubspot_created)

    def test_no_flag_true_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            svc.update_flags(self.rec.id)
        self.assertIn("At least one flag must be true", str(ctx.exception))

    def test_unknown_id_raises_not_found(self):
        unknown_id = "00000000-0000-0000-0000-000000000000"
        with self.assertRaises(svc.LeadStoreError) as ctx:
            svc.update_flags(unknown_id, sms_sent=True)
        self.assertEqual(ctx.exception.category, "not_found")

    def test_spy_on_update_asserts_false_never_written(self):
        svc.update_flags(self.rec.id, hubspot_created=True)
        # Check all recorded update dictionaries
        for update_dict in self.client.recorded_updates:
            self.assertNotIn(False, update_dict.values())
        self.assertEqual(self.client.recorded_updates[-1], {"hubspot_created": True})


if __name__ == "__main__":
    unittest.main()
