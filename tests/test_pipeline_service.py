"""
tests/test_pipeline_service.py

Automated test suite for Phase 9 pipeline service orchestration.
Follows strict rules:
- Uses ONLY unittest and unittest.mock.
- No network access, no .env file access.
- Environment-dependent tests use patch.dict(os.environ, {...}, clear=True).
- process_lead is ALWAYS called with lead_id and deps as keyword arguments.
- Uses subTest for table-driven scenarios.
- Inspects openai error constructors using inspect.signature and builds them
  with httpx.Request/httpx.Response.
"""

import importlib
import inspect
import json
import logging
import os
import unittest
from unittest.mock import patch

import httpx
import openai

from models import EmailPayload, LeadInput, LeadRecord, LeadScore, SMSPayload
import services.pipeline_service as pipeline_service
from services.hubspot_service import contact_from_record
from services.phone_service import PhoneNormalization, normalize_phone
from services.pipeline_service import PipelineDeps, process_lead


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_sample_lead(
    name: str = "Sarah Mitchell",
    email: str = "sarah.mitchell@example.com",
    phone: str = "+14155550192",
    property_type: str | None = "house",
    preferred_location: str | None = "Austin, TX",
    message: str = "Looking for a 3-bedroom house in Austin.",
    budget: str | None = "$450k-$550k",
    timeline: str | None = "within 3 months",
) -> LeadInput:
    """Build a valid LeadInput test instance with standard defaults."""
    return LeadInput(
        name=name,
        email=email,
        phone=phone,
        property_type=property_type,
        preferred_location=preferred_location,
        message=message,
        budget=budget,
        timeline=timeline,
    )


class FakeStore:
    """In-memory dictionary store keyed by id with first-write-wins semantics."""

    def __init__(self, call_log: list[str] | None = None) -> None:
        self.records: dict[str, LeadRecord] = {}
        self._next_store_exc: Exception | None = None
        self._next_flag_exc: Exception | None = None
        self.call_log = call_log

    def fail_next(self, exc: Exception) -> None:
        """Inject an exception to be raised on the next store_lead call."""
        self._next_store_exc = exc

    def fail_next_flag(self, exc: Exception) -> None:
        """Inject an exception to be raised on the next update_flags call."""
        self._next_flag_exc = exc

    def store_lead(self, record: LeadRecord) -> tuple[LeadRecord, bool]:
        """Store the lead record with first-write-wins semantics."""
        if self.call_log is not None:
            self.call_log.append("store")
        if self._next_store_exc is not None:
            exc = self._next_store_exc
            self._next_store_exc = None
            raise exc
        if record.id not in self.records:
            copy = record.model_copy(deep=True)
            self.records[record.id] = copy
            return copy.model_copy(deep=True), True

        existing = self.records[record.id]
        updates = {}
        for flag in ("email_sent", "sms_sent", "hubspot_created"):
            if getattr(record, flag) is True:
                updates[flag] = True
        if updates:
            existing = existing.model_copy(update=updates)
            self.records[record.id] = existing
        return existing.model_copy(deep=True), False

    def update_flags(
        self,
        lead_id: str,
        *,
        email_sent: bool = False,
        sms_sent: bool = False,
        hubspot_created: bool = False,
        **kwargs,
    ) -> LeadRecord:
        """Update delivery flags to True and return the updated copy."""
        flag_names = []
        if email_sent:
            flag_names.append("email_sent")
        if sms_sent:
            flag_names.append("sms_sent")
        if hubspot_created:
            flag_names.append("hubspot_created")
        for k, v in kwargs.items():
            if v:
                flag_names.append(k)

        flag_str = ",".join(flag_names)
        if self.call_log is not None:
            self.call_log.append(f"update_flags({flag_str})")

        if self._next_flag_exc is not None:
            exc = self._next_flag_exc
            self._next_flag_exc = None
            raise exc

        if lead_id not in self.records:
            raise pipeline_service.LeadStoreError(f"Lead {lead_id} not found", category="not_found")

        record = self.records[lead_id]
        updates = {}
        if email_sent:
            updates["email_sent"] = True
        if sms_sent:
            updates["sms_sent"] = True
        if hubspot_created:
            updates["hubspot_created"] = True
        for k, v in kwargs.items():
            if v is True and hasattr(record, k):
                updates[k] = True

        updated = record.model_copy(update=updates)
        self.records[lead_id] = updated
        return updated.model_copy(deep=True)


def make_deps(
    call_log: list[str] | None = None,
    store: FakeStore | None = None,
    **overrides,
) -> PipelineDeps:
    """Build a PipelineDeps instance with fakes logging calls and safe defaults."""
    active_store = store if store is not None else FakeStore(call_log=call_log)

    def default_score_lead(lead: LeadInput) -> LeadScore:
        if call_log is not None:
            call_log.append("score")
        return LeadScore(
            score=87,
            reasoning="Qualified buyer with clear preferences.",
            suggested_action="Contact today.",
            priority_level="urgent",
        )

    def default_generate_email(lead: LeadInput, score: LeadScore) -> EmailPayload:
        if call_log is not None:
            call_log.append("generate_email")
        return EmailPayload(
            to_email=lead.email,
            to_name=lead.name,
            subject="Your Property Search",
            body="Hello Sarah, we found homes for you.",
            lead_score=score.score,
        )

    def default_send_email(payload: object) -> dict:
        if call_log is not None:
            call_log.append("send_email")
        return {"dry_run": False, "message_id": "m1"}

    def default_generate_sms(lead: LeadInput, score: LeadScore) -> SMSPayload:
        if call_log is not None:
            call_log.append("generate_sms")
        return SMSPayload(
            to_phone=lead.phone,
            message="Hello from Real Estate Team.",
            lead_score=score.score,
        )

    def default_send_sms(payload: object) -> dict:
        if call_log is not None:
            call_log.append("send_sms")
        return {"dry_run": False, "message_id": "SM1"}

    def default_upsert_contact(contact: object) -> dict:
        if call_log is not None:
            call_log.append("upsert_contact")
        return {"dry_run": False, "contact_id": "512"}

    deps_dict = {
        "normalize_phone": normalize_phone,
        "score_lead": default_score_lead,
        "store_lead": active_store.store_lead,
        "update_flags": active_store.update_flags,
        "generate_email": default_generate_email,
        "send_email": default_send_email,
        "generate_sms": default_generate_sms,
        "send_sms": default_send_sms,
        "contact_from_record": contact_from_record,
        "upsert_contact": default_upsert_contact,
    }
    deps_dict.update(overrides)
    return PipelineDeps(**deps_dict)


def make_rate_limit_error(message: str = "Rate limit exceeded") -> openai.RateLimitError:
    """Build an openai.RateLimitError with httpx.Request/Response."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    return openai.RateLimitError(message, response=resp, body=None)


def make_api_connection_error(message: str = "Connection error.") -> openai.APIConnectionError:
    """Build an openai.APIConnectionError with httpx.Request."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.APIConnectionError(message=message, request=req)


def make_authentication_error(message: str = "Invalid API key") -> openai.AuthenticationError:
    """Build an openai.AuthenticationError with httpx.Request/Response."""
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    return openai.AuthenticationError(message, response=resp, body=None)


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

class TestPipelineServiceA_ImportSafety(unittest.TestCase):
    """A. Import safety, encoding, and source hygiene."""

    def test_import_with_empty_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(pipeline_service)

    def test_source_is_pure_ascii(self):
        source_path = inspect.getfile(pipeline_service)
        with open(source_path, "rb") as f:
            raw = f.read()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"pipeline_service.py is not pure ASCII: {exc}")

    def test_source_contains_neither_os_getenv_nor_os_environ(self):
        source_path = inspect.getfile(pipeline_service)
        with open(source_path, "r", encoding="ascii") as f:
            content = f.read()
        self.assertNotIn("os.getenv", content)
        self.assertNotIn("os.environ", content)

    def test_openai_error_signatures_and_construction(self):
        sig_rle = inspect.signature(openai.RateLimitError.__init__)
        sig_ace = inspect.signature(openai.APIConnectionError.__init__)
        sig_auth = inspect.signature(openai.AuthenticationError.__init__)

        self.assertIn("response", sig_rle.parameters)
        self.assertIn("request", sig_ace.parameters)
        self.assertIn("response", sig_auth.parameters)

        err_rle = make_rate_limit_error("rate limit")
        err_ace = make_api_connection_error("conn err")
        err_auth = make_authentication_error("auth err")

        self.assertIsInstance(err_rle, openai.RateLimitError)
        self.assertIsInstance(err_ace, openai.APIConnectionError)
        self.assertIsInstance(err_auth, openai.AuthenticationError)
        self.assertIsInstance(err_auth, openai.APIStatusError)
        self.assertEqual(err_auth.status_code, 401)


class TestPipelineServiceB_HappyPath(unittest.TestCase):
    """B. Happy path, hot lead, all live-like, and warm lead."""

    def test_happy_path_hot_lead(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        deps = make_deps(call_log=call_log, store=store)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        expected_order = [
            "score",
            "store",
            "generate_email",
            "send_email",
            "update_flags(email_sent)",
            "generate_sms",
            "send_sms",
            "update_flags(sms_sent)",
            "upsert_contact",
            "update_flags(hubspot_created)",
        ]
        self.assertEqual(call_log, expected_order)

        step_names = [s.name for s in result.steps]
        self.assertEqual(step_names, ["intake", "score", "store", "email", "sms", "crm"])

        step_dict = {s.name: s for s in result.steps}
        for step_name, ext_id in [("email", "m1"), ("sms", "SM1"), ("crm", "512")]:
            step = step_dict[step_name]
            self.assertEqual(step.status, "succeeded")
            self.assertEqual(step.external_id, ext_id)
            self.assertTrue(step.flag_recorded)

        stored = store.records[result.lead_id]
        self.assertTrue(stored.email_sent)
        self.assertTrue(stored.sms_sent)
        self.assertTrue(stored.hubspot_created)

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.created)
        self.assertFalse(result.retry_steps)
        self.assertFalse(result.to_dict()["retry"]["recommended"])

        dumped = json.dumps(result.to_dict())
        self.assertIsInstance(dumped, str)

    def test_happy_path_warm_lead_behaves_the_same(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)

        def score_warm(lead: LeadInput) -> LeadScore:
            call_log.append("score")
            return LeadScore(
                score=50,
                reasoning="Moderate budget and standard timeline.",
                suggested_action="Send info pack.",
                priority_level="normal",
            )

        deps = make_deps(call_log=call_log, store=store, score_lead=score_warm)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        expected_order = [
            "score",
            "store",
            "generate_email",
            "send_email",
            "update_flags(email_sent)",
            "generate_sms",
            "send_sms",
            "update_flags(sms_sent)",
            "upsert_contact",
            "update_flags(hubspot_created)",
        ]
        self.assertEqual(call_log, expected_order)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.category, "warm")
        self.assertEqual(result.score, 50)
        self.assertTrue(result.created)

        stored = store.records[result.lead_id]
        self.assertTrue(stored.email_sent)
        self.assertTrue(stored.sms_sent)
        self.assertTrue(stored.hubspot_created)


class TestPipelineServiceC_ColdLead(unittest.TestCase):
    """C. Cold lead skipping outbound actions."""

    def test_cold_lead_skips_all_outbound(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)

        def score_cold(lead: LeadInput) -> LeadScore:
            call_log.append("score")
            return LeadScore(
                score=20,
                reasoning="Low budget, indefinite timeline.",
                suggested_action="Newsletter only.",
                priority_level="low",
            )

        deps = make_deps(call_log=call_log, store=store, score_lead=score_cold)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        self.assertEqual(call_log, ["score", "store"])
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.persisted)
        self.assertTrue(result.created)
        self.assertEqual(result.category, "cold")

        step_dict = {s.name: s for s in result.steps}
        for name in ("email", "sms", "crm"):
            step = step_dict[name]
            self.assertEqual(step.status, "skipped")
            self.assertEqual(step.reason, "cold_lead")

        stored = store.records[result.lead_id]
        self.assertFalse(stored.email_sent)
        self.assertFalse(stored.sms_sent)
        self.assertFalse(stored.hubspot_created)


class TestPipelineServiceD_DryRun(unittest.TestCase):
    """D. Dry-run and non-bool dry_run handling, table-driven."""

    def test_dry_run_non_bool_skips_and_leaves_flags_false(self):
        cases = [
            ("dry_run_True", {"dry_run": True, "message_id": "m1"}),
            ("no_dry_run_key", {"message_id": "m1"}),
            ("dry_run_None", {"dry_run": None, "message_id": "m1"}),
            ("dry_run_0", {"dry_run": 0, "message_id": "m1"}),
        ]

        for label, resp_payload in cases:
            with self.subTest(case=label):
                call_log: list[str] = []
                store = FakeStore(call_log=call_log)
                deps = make_deps(
                    call_log=call_log,
                    store=store,
                    send_email=lambda _p: resp_payload,
                    send_sms=lambda _p: resp_payload,
                    upsert_contact=lambda _p: resp_payload,
                )
                lead = make_sample_lead()

                result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                self.assertEqual(result.status, "completed")
                step_dict = {s.name: s for s in result.steps}
                for name in ("email", "sms", "crm"):
                    step = step_dict[name]
                    self.assertEqual(step.status, "skipped")
                    self.assertEqual(step.reason, "dry_run")

                for entry in call_log:
                    self.assertNotIn("update_flags", entry)

                stored = store.records[result.lead_id]
                self.assertFalse(stored.email_sent)
                self.assertFalse(stored.sms_sent)
                self.assertFalse(stored.hubspot_created)

    def test_non_dict_send_response_fails_step_unexpected(self):
        invalid_responses = [
            ("list", ["not", "a", "dict"]),
            ("string", "unexpected response text"),
            ("int", 42),
        ]

        for label, bad_resp in invalid_responses:
            with self.subTest(case=label):
                call_log: list[str] = []
                store = FakeStore(call_log=call_log)
                deps = make_deps(
                    call_log=call_log,
                    store=store,
                    send_email=lambda _p: bad_resp,
                )
                lead = make_sample_lead()

                result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                self.assertEqual(result.status, "partial")
                step_dict = {s.name: s for s in result.steps}

                email_step = step_dict["email"]
                self.assertEqual(email_step.status, "failed")
                self.assertEqual(email_step.reason, "unexpected")
                self.assertFalse(email_step.retryable)

                self.assertEqual(step_dict["sms"].status, "succeeded")
                self.assertEqual(step_dict["crm"].status, "succeeded")

                stored = store.records[result.lead_id]
                self.assertFalse(stored.email_sent)
                self.assertTrue(stored.sms_sent)
                self.assertTrue(stored.hubspot_created)

    def test_dry_run_false_missing_or_blank_external_id(self):
        bad_ids = [
            ("none", None),
            ("empty", ""),
            ("whitespace", "   "),
        ]

        for label, bad_id in bad_ids:
            with self.subTest(case=label):
                call_log: list[str] = []
                store = FakeStore(call_log=call_log)
                deps = make_deps(
                    call_log=call_log,
                    store=store,
                    send_email=lambda _p: {"dry_run": False, "message_id": bad_id},
                )
                lead = make_sample_lead()

                result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                step_dict = {s.name: s for s in result.steps}
                email_step = step_dict["email"]
                self.assertEqual(email_step.status, "failed")
                self.assertEqual(email_step.reason, "missing_external_id")
                self.assertFalse(email_step.retryable)

                self.assertEqual(step_dict["sms"].status, "succeeded")
                self.assertEqual(step_dict["crm"].status, "succeeded")
                self.assertEqual(result.status, "partial")

                stored = store.records[result.lead_id]
                self.assertFalse(stored.email_sent)
                self.assertTrue(stored.sms_sent)
                self.assertTrue(stored.hubspot_created)


class TestPipelineServiceE_AlreadyDone(unittest.TestCase):
    """E. Stored delivery flags acting as idempotency guards."""

    def test_stored_email_sent_true_skips_email_only(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        lead = make_sample_lead()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

        seed_record = LeadRecord(
            id=lead_id,
            lead=lead,
            score=LeadScore(score=87, reasoning="Good", suggested_action="Act", priority_level="urgent"),
            email_sent=True,
            sms_sent=False,
            hubspot_created=False,
        )
        store.records[lead_id] = seed_record

        deps = make_deps(call_log=call_log, store=store)
        result = process_lead(lead, lead_id=lead_id, deps=deps)

        self.assertNotIn("generate_email", call_log)
        self.assertNotIn("send_email", call_log)
        self.assertIn("generate_sms", call_log)
        self.assertIn("send_sms", call_log)
        self.assertIn("upsert_contact", call_log)

        step_dict = {s.name: s for s in result.steps}
        self.assertEqual(step_dict["email"].status, "skipped")
        self.assertEqual(step_dict["email"].reason, "already_done")
        self.assertEqual(step_dict["sms"].status, "succeeded")
        self.assertEqual(step_dict["crm"].status, "succeeded")
        self.assertEqual(result.status, "completed")

    def test_all_three_flags_true_skips_all_outbound(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        lead = make_sample_lead()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

        seed_record = LeadRecord(
            id=lead_id,
            lead=lead,
            score=LeadScore(score=87, reasoning="Good", suggested_action="Act", priority_level="urgent"),
            email_sent=True,
            sms_sent=True,
            hubspot_created=True,
        )
        store.records[lead_id] = seed_record

        deps = make_deps(call_log=call_log, store=store)
        result = process_lead(lead, lead_id=lead_id, deps=deps)

        self.assertEqual(call_log, ["score", "store"])
        step_dict = {s.name: s for s in result.steps}
        for name in ("email", "sms", "crm"):
            self.assertEqual(step_dict[name].status, "skipped")
            self.assertEqual(step_dict[name].reason, "already_done")
        self.assertEqual(result.status, "completed")


class TestPipelineServiceF_DatabaseWins(unittest.TestCase):
    """F. Database authority and first-write-wins semantics."""

    def test_first_hot_second_cold_routes_as_hot_already_done(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        lead = make_sample_lead()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

        # First run: hot score
        deps1 = make_deps(call_log=call_log, store=store)
        res1 = process_lead(lead, lead_id=lead_id, deps=deps1)
        self.assertEqual(res1.status, "completed")
        self.assertTrue(res1.created)
        self.assertEqual(res1.category, "hot")

        call_log.clear()

        # Second run with cold score
        def score_cold(l: LeadInput) -> LeadScore:
            call_log.append("score")
            return LeadScore(score=20, reasoning="Cold", suggested_action="Wait", priority_level="low")

        deps2 = make_deps(call_log=call_log, store=store, score_lead=score_cold)
        res2 = process_lead(lead, lead_id=lead_id, deps=deps2)

        self.assertFalse(res2.created)
        self.assertEqual(res2.category, "hot")
        self.assertEqual(res2.score, 87)
        self.assertEqual(res2.status, "completed")

        step_dict = {s.name: s for s in res2.steps}
        for name in ("email", "sms", "crm"):
            self.assertEqual(step_dict[name].status, "skipped")
            self.assertEqual(step_dict[name].reason, "already_done")

        self.assertNotIn("generate_email", call_log)
        self.assertNotIn("send_email", call_log)
        self.assertNotIn("generate_sms", call_log)
        self.assertNotIn("send_sms", call_log)
        self.assertNotIn("upsert_contact", call_log)

    def test_first_cold_second_hot_stays_cold(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        lead = make_sample_lead()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

        # First run: cold score
        def score_cold(l: LeadInput) -> LeadScore:
            call_log.append("score")
            return LeadScore(score=20, reasoning="Cold", suggested_action="Wait", priority_level="low")

        deps1 = make_deps(call_log=call_log, store=store, score_lead=score_cold)
        res1 = process_lead(lead, lead_id=lead_id, deps=deps1)
        self.assertEqual(res1.category, "cold")
        self.assertTrue(res1.created)

        call_log.clear()

        # Second run: hot score
        deps2 = make_deps(call_log=call_log, store=store)
        res2 = process_lead(lead, lead_id=lead_id, deps=deps2)

        self.assertFalse(res2.created)
        self.assertEqual(res2.category, "cold")
        self.assertEqual(res2.score, 20)

        step_dict = {s.name: s for s in res2.steps}
        for name in ("email", "sms", "crm"):
            self.assertEqual(step_dict[name].status, "skipped")
            self.assertEqual(step_dict[name].reason, "cold_lead")


class TestPipelineServiceG_StepIsolation(unittest.TestCase):
    """G. Outbound step error classification, retryability, and isolation."""

    def test_outbound_exceptions_table_driven(self):
        cases = [
            ("email", "recipient", pipeline_service.EmailSendError("bad email", category="recipient"), False),
            ("email", "config", pipeline_service.EmailSendError("no auth", category="config"), False),
            ("email", "transient", pipeline_service.EmailSendError("timeout", category="transient"), True),
            ("sms", "recipient", pipeline_service.SMSSendError("unreachable", category="recipient"), False),
            ("sms", "config", pipeline_service.SMSSendError("bad SID", category="config"), False),
            ("sms", "transient", pipeline_service.SMSSendError("carrier err", category="transient"), True),
            ("crm", "config", pipeline_service.CRMError("bad token", category="config"), False),
            ("crm", "invalid", pipeline_service.CRMError("invalid prop", category="invalid"), False),
            ("crm", "conflict", pipeline_service.CRMError("contact collision", category="conflict"), False),
            ("crm", "transient", pipeline_service.CRMError("503 down", category="transient"), True),
        ]

        for step_target, category, exc, expected_retryable in cases:
            with self.subTest(step=step_target, category=category):
                store = FakeStore()
                overrides = {}
                if step_target == "email":
                    overrides["send_email"] = lambda _p, e=exc: (_ for _ in ()).throw(e)
                elif step_target == "sms":
                    overrides["send_sms"] = lambda _p, e=exc: (_ for _ in ()).throw(e)
                elif step_target == "crm":
                    overrides["upsert_contact"] = lambda _p, e=exc: (_ for _ in ()).throw(e)

                deps = make_deps(store=store, **overrides)
                lead = make_sample_lead()

                result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                self.assertEqual(result.status, "partial")
                step_dict = {s.name: s for s in result.steps}

                failing_step = step_dict[step_target]
                self.assertEqual(failing_step.status, "failed")
                self.assertEqual(failing_step.reason, category)
                self.assertEqual(failing_step.retryable, expected_retryable)

                # Confirm the other two outbound steps succeeded
                other_steps = [s for s in ("email", "sms", "crm") if s != step_target]
                for other in other_steps:
                    self.assertEqual(step_dict[other].status, "succeeded")

                if expected_retryable:
                    self.assertEqual(result.retry_steps, [step_target])
                    self.assertTrue(result.to_dict()["retry"]["recommended"])
                else:
                    self.assertEqual(result.retry_steps, [])
                    self.assertFalse(result.to_dict()["retry"]["recommended"])

    def test_precheck_exception_fails_step_without_blocking_other_steps(self):
        call_count = [0]

        def phone_raiser(raw: object) -> PhoneNormalization:
            call_count[0] += 1
            if call_count[0] == 1:
                return PhoneNormalization(e164="+14155550192", reason="international")
            raise RuntimeError("precheck failure")

        store = FakeStore()
        deps = make_deps(store=store, normalize_phone=phone_raiser)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        self.assertEqual(result.status, "partial")
        step_dict = {s.name: s for s in result.steps}

        self.assertEqual(step_dict["email"].status, "succeeded")
        self.assertEqual(step_dict["sms"].status, "failed")
        self.assertEqual(step_dict["sms"].reason, "config")
        self.assertFalse(step_dict["sms"].retryable)
        self.assertEqual(step_dict["crm"].status, "succeeded")


class TestPipelineServiceH_LLMFailures(unittest.TestCase):
    """H. LLM generation failures in generate_email."""

    def test_llm_failures_in_generate_email(self):
        cases = [
            (
                "rate_limit",
                make_rate_limit_error("OpenAI rate limit"),
                "transient",
                True,
                lambda msg: True,
            ),
            (
                "api_connection",
                make_api_connection_error("OpenAI connect error"),
                "transient",
                True,
                lambda msg: True,
            ),
            (
                "authentication",
                make_authentication_error("OpenAI auth error"),
                "config",
                False,
                lambda msg: True,
            ),
            (
                "value_error",
                ValueError("bad model output formatting"),
                "model_output",
                True,
                lambda msg: True,
            ),
            (
                "runtime_error",
                RuntimeError("missing OPENROUTER_API_KEY"),
                "config",
                False,
                lambda msg: "OPENROUTER_API_KEY" in msg,
            ),
            (
                "generic_exception",
                Exception("sarah.mitchell@example.com boom"),
                "unexpected",
                False,
                lambda msg: "boom" not in msg and "sarah.mitchell@example.com" not in msg,
            ),
        ]

        for label, exc, expected_reason, expected_retryable, msg_check in cases:
            with self.subTest(case=label):
                store = FakeStore()

                def generate_failing(_lead: LeadInput, _score: LeadScore):
                    raise exc

                deps = make_deps(store=store, generate_email=generate_failing)
                lead = make_sample_lead()

                result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                self.assertEqual(result.status, "partial")
                step_dict = {s.name: s for s in result.steps}

                email_step = step_dict["email"]
                self.assertEqual(email_step.status, "failed")
                self.assertEqual(email_step.reason, expected_reason)
                self.assertEqual(email_step.retryable, expected_retryable)
                self.assertIsNotNone(email_step.error)
                self.assertTrue(msg_check(email_step.error.message))

                self.assertEqual(step_dict["sms"].status, "succeeded")
                self.assertEqual(step_dict["crm"].status, "succeeded")

                if expected_retryable:
                    self.assertEqual(result.retry_steps, ["email"])
                else:
                    self.assertEqual(result.retry_steps, [])


class TestPipelineServiceI_ScoreAndStoreFailures(unittest.TestCase):
    """I. Early pipeline failures in score and store steps."""

    def test_score_failure_transient(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)

        def score_fail(_l: LeadInput):
            call_log.append("score")
            raise make_rate_limit_error("AI rate limit")

        deps = make_deps(call_log=call_log, store=store, score_lead=score_fail)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.persisted)
        self.assertEqual(result.failed_step, "score")
        self.assertEqual(result.retry_steps, ["score"])
        self.assertTrue(result.to_dict()["retry"]["recommended"])
        self.assertNotIn("store", call_log)

        step_dict = {s.name: s for s in result.steps}
        self.assertEqual(step_dict["score"].status, "failed")
        self.assertEqual(step_dict["score"].reason, "transient")
        for name in ("store", "email", "sms", "crm"):
            self.assertEqual(step_dict[name].status, "skipped")
            self.assertEqual(step_dict[name].reason, "upstream_failed")

    def test_score_failure_config_retry_not_recommended(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)

        def score_fail(_l: LeadInput):
            call_log.append("score")
            raise make_authentication_error("bad api key")

        deps = make_deps(call_log=call_log, store=store, score_lead=score_fail)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        self.assertEqual(result.status, "failed")
        self.assertFalse(result.persisted)
        self.assertEqual(result.failed_step, "score")
        self.assertEqual(result.retry_steps, [])
        self.assertFalse(result.to_dict()["retry"]["recommended"])
        self.assertNotIn("store", call_log)

    def test_store_failure_transient_and_config(self):
        cases = [
            ("transient", pipeline_service.LeadStoreError("db timeout", category="transient"), True),
            ("config", pipeline_service.LeadStoreError("bad schema", category="config"), False),
        ]

        for label, exc, expected_retry in cases:
            with self.subTest(case=label):
                call_log: list[str] = []
                store = FakeStore(call_log=call_log)
                store.fail_next(exc)
                deps = make_deps(call_log=call_log, store=store)
                lead = make_sample_lead()

                result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                self.assertEqual(result.status, "failed")
                self.assertFalse(result.persisted)
                self.assertEqual(result.failed_step, "store")
                step_dict = {s.name: s for s in result.steps}
                self.assertEqual(step_dict["store"].reason, label)

                for name in ("email", "sms", "crm"):
                    self.assertEqual(step_dict[name].status, "skipped")
                    self.assertEqual(step_dict[name].reason, "upstream_failed")

                if expected_retry:
                    self.assertEqual(result.retry_steps, ["store"])
                    self.assertTrue(result.to_dict()["retry"]["recommended"])
                else:
                    self.assertEqual(result.retry_steps, [])
                    self.assertFalse(result.to_dict()["retry"]["recommended"])

    def test_store_failure_after_record_construction_and_invalid_data(self):
        # Case 1: store_lead raises after construction
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        store.fail_next(RuntimeError("disk write failed"))
        deps = make_deps(call_log=call_log, store=store)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed_step, "store")
        self.assertFalse(result.persisted)

        # Case 2: store_lead raises because lead/score data itself is invalid
        store_invalid = FakeStore()
        store_invalid.fail_next(pipeline_service.LeadStoreError("invalid score data in store", category="invalid"))
        deps_invalid = make_deps(store=store_invalid)
        result_inv = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps_invalid)

        self.assertEqual(result_inv.status, "failed")
        self.assertEqual(result_inv.failed_step, "store")
        self.assertFalse(result_inv.persisted)


class TestPipelineServiceJ_FlagUpdateFailure(unittest.TestCase):
    """J. Flag update failure, retry attempts, and repair payload."""

    def test_flag_update_always_fails(self):
        call_log: list[str] = []
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        store = FakeStore(call_log=call_log)

        def failing_email_update_flags(lid: str, **kwargs):
            if kwargs.get("email_sent"):
                call_log.append("update_flags(email_sent)")
                raise pipeline_service.LeadStoreError("database connection broken", category="transient")
            return store.update_flags(lid, **kwargs)

        deps = make_deps(
            call_log=call_log,
            store=store,
            update_flags=failing_email_update_flags,
        )
        lead = make_sample_lead()

        with self.assertLogs(pipeline_service.logger, level="ERROR") as cm:
            result = process_lead(lead, lead_id=lead_id, deps=deps)

        self.assertEqual(result.status, "partial")
        self.assertFalse(result.retry_steps)
        self.assertFalse(result.to_dict()["retry"]["recommended"])

        step_dict = {s.name: s for s in result.steps}
        email_step = step_dict["email"]
        self.assertEqual(email_step.status, "failed")
        self.assertEqual(email_step.reason, "flag_update_failed")
        self.assertFalse(email_step.retryable)
        self.assertEqual(email_step.external_id, "m1")
        self.assertFalse(email_step.flag_recorded)

        self.assertEqual(result.repair, [
            {
                "action": f"PATCH /leads/{lead_id}/flags",
                "body": {"email_sent": True},
            }
        ])

        # Check email's update_flags was attempted exactly 2 times
        email_flag_attempts = [entry for entry in call_log if "email_sent" in entry]
        self.assertEqual(len(email_flag_attempts), 2)

        # Confirm sms and crm still ran
        self.assertIn("send_sms", call_log)
        self.assertIn("upsert_contact", call_log)

        # Confirm ERROR log mentions "not flagged"
        error_logs = [record.getMessage() for record in cm.records if record.levelno == logging.ERROR]
        self.assertTrue(any("not flagged" in msg for msg in error_logs))

    def test_flag_update_first_fails_second_succeeds(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

        deps = make_deps(call_log=call_log, store=store)
        lead = make_sample_lead()

        # Seed the first update_flags call to fail
        store.fail_next_flag(pipeline_service.LeadStoreError("transient error", category="transient"))

        result = process_lead(lead, lead_id=lead_id, deps=deps)

        self.assertEqual(result.status, "completed")
        step_dict = {s.name: s for s in result.steps}
        email_step = step_dict["email"]
        self.assertEqual(email_step.status, "succeeded")
        self.assertTrue(email_step.flag_recorded)
        self.assertEqual(email_step.external_id, "m1")
        self.assertEqual(result.repair, [])


class TestPipelineServiceK_Intake(unittest.TestCase):
    """K. Intake ID canonicalisation, validation, and phone handling."""

    def test_uppercase_hyphenless_id_canonicalised(self):
        store = FakeStore()
        deps = make_deps(store=store)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3FA85F6457174562B3FC2C963F66AFA6", deps=deps)

        self.assertEqual(result.lead_id, "3fa85f64-5717-4562-b3fc-2c963f66afa6")
        self.assertFalse(result.idempotency_key_generated)

    def test_invalid_id_raises_value_error_before_collaborators(self):
        call_log: list[str] = []
        store = FakeStore(call_log=call_log)
        deps = make_deps(call_log=call_log, store=store)
        lead = make_sample_lead()

        with self.assertRaises(ValueError):
            process_lead(lead, lead_id="abc", deps=deps)

        self.assertEqual(call_log, [])

    def test_missing_id_generates_uuid_with_flag(self):
        store = FakeStore()
        deps = make_deps(store=store)
        lead = make_sample_lead()

        result = process_lead(lead, deps=deps)

        self.assertTrue(result.idempotency_key_generated)
        self.assertEqual(len(result.lead_id), 36)
        self.assertIn("-", result.lead_id)

    def test_phone_normalization_with_country_code(self):
        with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": "US"}, clear=True):
            captured_scored_lead: list[LeadInput] = []

            def score_inspector(l: LeadInput) -> LeadScore:
                captured_scored_lead.append(l)
                return LeadScore(score=87, reasoning="Good", suggested_action="Act", priority_level="urgent")

            store = FakeStore()
            deps = make_deps(store=store, score_lead=score_inspector)
            lead = make_sample_lead(phone="(415) 555-0192")

            result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

            self.assertEqual(captured_scored_lead[0].phone, "+14155550192")
            step_dict = {s.name: s for s in result.steps}
            self.assertEqual(step_dict["intake"].reason, "normalized")
            stored = store.records[result.lead_id]
            self.assertEqual(stored.lead.phone, "+14155550192")

    def test_phone_without_region_stores_raw_and_skips_sms(self):
        with patch.dict(os.environ, {}, clear=True):
            store = FakeStore()
            deps = make_deps(store=store)
            lead = make_sample_lead(phone="(415) 555-0192")

            result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

            stored = store.records[result.lead_id]
            self.assertEqual(stored.lead.phone, "(415) 555-0192")

            step_dict = {s.name: s for s in result.steps}
            self.assertEqual(step_dict["intake"].reason, "needs_country_code")
            self.assertEqual(step_dict["sms"].status, "skipped")
            self.assertEqual(step_dict["sms"].reason, "needs_country_code")
            self.assertEqual(step_dict["email"].status, "succeeded")
            self.assertEqual(step_dict["crm"].status, "succeeded")

    def test_phone_invalid_skips_sms(self):
        with patch.dict(os.environ, {}, clear=True):
            store = FakeStore()
            deps = make_deps(store=store)
            lead = make_sample_lead(phone="abcdefg")

            result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

            step_dict = {s.name: s for s in result.steps}
            self.assertEqual(step_dict["sms"].status, "skipped")
            self.assertEqual(step_dict["sms"].reason, "invalid_phone")


class TestPipelineServiceL_RenormalisationAtSMSTime(unittest.TestCase):
    """L. Phone re-normalisation during subsequent retry execution."""

    def test_renormalisation_at_sms_time(self):
        store = FakeStore()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        lead = make_sample_lead(phone="(415) 555-0192")

        # Run 1: without DEFAULT_COUNTRY_CODE
        with patch.dict(os.environ, {}, clear=True):
            deps1 = make_deps(store=store)
            res1 = process_lead(lead, lead_id=lead_id, deps=deps1)

            stored1 = store.records[lead_id]
            self.assertEqual(stored1.lead.phone, "(415) 555-0192")
            self.assertTrue(stored1.email_sent)
            self.assertFalse(stored1.sms_sent)
            self.assertTrue(stored1.hubspot_created)
            step_dict1 = {s.name: s for s in res1.steps}
            self.assertEqual(step_dict1["sms"].status, "skipped")
            self.assertEqual(step_dict1["sms"].reason, "needs_country_code")

        # Run 2: with DEFAULT_COUNTRY_CODE configured
        with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": "US"}, clear=True):
            captured_sms_lead: list[LeadInput] = []

            def generate_sms_capture(l: LeadInput, _score: LeadScore) -> SMSPayload:
                captured_sms_lead.append(l)
                return SMSPayload(to_phone=l.phone, message="Hello!", lead_score=87)

            deps2 = make_deps(store=store, generate_sms=generate_sms_capture)
            res2 = process_lead(lead, lead_id=lead_id, deps=deps2)

            self.assertEqual(len(captured_sms_lead), 1)
            self.assertEqual(captured_sms_lead[0].phone, "+14155550192")

            step_dict2 = {s.name: s for s in res2.steps}
            self.assertEqual(step_dict2["email"].status, "skipped")
            self.assertEqual(step_dict2["email"].reason, "already_done")
            self.assertEqual(step_dict2["sms"].status, "succeeded")
            self.assertTrue(step_dict2["sms"].flag_recorded)
            self.assertEqual(step_dict2["crm"].status, "skipped")
            self.assertEqual(step_dict2["crm"].reason, "already_done")

            stored2 = store.records[lead_id]
            self.assertTrue(stored2.sms_sent)


class TestPipelineServiceM_Privacy(unittest.TestCase):
    """M. Privacy preservation: customer data must never leak into logs or report."""

    def test_canary_values_never_leak(self):
        canaries = [
            "Zed Canary",
            "canary.lead@example.com",
            "+14155550142",
            "canary message text",
        ]
        lead = make_sample_lead(
            name="Zed Canary",
            email="canary.lead@example.com",
            phone="+14155550142",
            message="canary message text",
        )

        test_scenarios = [
            ("success", {}),
            ("score_failure", {"score_lead": lambda _l: (_ for _ in ()).throw(RuntimeError("fail"))}),
            ("store_failure", {"store_lead": lambda _r: (_ for _ in ()).throw(pipeline_service.LeadStoreError("err", category="transient"))}),
            ("generate_email_fail", {"generate_email": lambda _l, _s: (_ for _ in ()).throw(ValueError("bad output"))}),
            ("send_email_fail", {"send_email": lambda _p: (_ for _ in ()).throw(pipeline_service.EmailSendError("err", category="transient"))}),
            ("flag_update_fail", {"update_flags": lambda _lid, **_kw: (_ for _ in ()).throw(pipeline_service.LeadStoreError("err", category="transient"))}),
        ]

        for label, overrides in test_scenarios:
            with self.subTest(scenario=label):
                store = FakeStore()
                deps = make_deps(store=store, **overrides)

                with self.assertLogs(pipeline_service.logger) as cm:
                    # Provide an info log so assertLogs doesn't fail on quiet happy-path
                    pipeline_service.logger.info("pipeline test run: scenario=%s", label)
                    result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

                serialized_report = json.dumps(result.to_dict())
                for canary in canaries:
                    self.assertNotIn(canary, serialized_report)

                for log_record in cm.records:
                    for canary in canaries:
                        self.assertNotIn(canary, log_record.getMessage())


class TestPipelineServiceN_Logging(unittest.TestCase):
    """N. Failed step warning logging contract."""

    def test_failed_step_produces_exactly_one_warning_with_expected_metadata(self):
        store = FakeStore()
        deps = make_deps(
            store=store,
            send_email=lambda _p: (_ for _ in ()).throw(pipeline_service.EmailSendError("network reset", category="transient")),
        )
        lead = make_sample_lead()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

        with self.assertLogs("services.pipeline_service", level="WARNING") as cm:
            result = process_lead(lead, lead_id=lead_id, deps=deps)

        self.assertEqual(result.status, "partial")
        warning_records = [rec for rec in cm.records if rec.levelno == logging.WARNING]
        self.assertEqual(len(warning_records), 1)

        msg = warning_records[0].getMessage()
        self.assertIn(lead_id, msg)
        self.assertIn("email", msg)
        self.assertIn("transient", msg)
        self.assertIn("EmailSendError", msg)


class TestPipelineServiceO_Duration(unittest.TestCase):
    """O. Step execution measurement non-negativity."""

    def test_every_step_has_integer_duration_gte_zero(self):
        store = FakeStore()
        deps = make_deps(store=store)
        lead = make_sample_lead()

        result = process_lead(lead, lead_id="3fa85f64-5717-4562-b3fc-2c963f66afa6", deps=deps)

        self.assertEqual(len(result.steps), 6)
        for step in result.steps:
            self.assertIsInstance(step.duration_ms, int)
            self.assertGreaterEqual(step.duration_ms, 0)


if __name__ == "__main__":
    unittest.main()
