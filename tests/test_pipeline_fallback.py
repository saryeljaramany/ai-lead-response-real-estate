"""
tests/test_pipeline_fallback.py

Unit tests for the opt-in last-resort fallback score in services/pipeline_service.py.
Follows all project testing standards:
- Pure ASCII, LF only, no BOM.
- Standard library unittest plus project models and services.
- No network, no environment file access.
"""

import json
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from models import LeadInput, LeadRecord, LeadScore
from services.fallback_scoring import FALLBACK_REASONING_PREFIX, fallback_score
import services.pipeline_service as pipeline_service
from services.pipeline_service import PipelineDeps, PipelineResult, process_lead, run_pipeline
from tests.test_pipeline_service import (
    FakeStore,
    make_api_connection_error,
    make_authentication_error,
    make_deps,
    make_rate_limit_error,
    make_sample_lead,
)


class TestPipelineFallbackScoring(unittest.TestCase):
    """Test suite for opt-in fallback scoring in run_pipeline."""

    def test_default_allow_fallback_false_fails_on_rate_limit(self):
        """Default allow_fallback_score=False fails when AI score raises RateLimitError."""
        fallback_spy = MagicMock()
        call_log = []
        store = FakeStore(call_log=call_log)
        deps = make_deps(
            call_log=call_log,
            store=store,
            score_lead=MagicMock(side_effect=make_rate_limit_error()),
            fallback_score=fallback_spy,
        )
        lead = make_sample_lead()

        result = run_pipeline(lead, deps=deps, allow_fallback_score=False)

        fallback_spy.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed_step, "score")
        self.assertFalse(result.persisted)
        self.assertNotIn("store", call_log)
        self.assertEqual(len(store.records), 0)

    def test_allow_fallback_true_transient_and_value_error_succeeds(self):
        """allow_fallback_score=True recovers from transient and model_output errors."""
        errors = [
            make_rate_limit_error("rate limited"),
            make_api_connection_error("conn failed"),
            ValueError("unusable output"),
        ]
        for exc in errors:
            with self.subTest(exc=type(exc).__name__):
                store = FakeStore()
                deps = make_deps(
                    store=store,
                    score_lead=MagicMock(side_effect=exc),
                )
                lead = make_sample_lead()
                result = run_pipeline(lead, deps=deps, allow_fallback_score=True)

                self.assertEqual(result.status, "completed")
                score_step = next(s for s in result.steps if s.name == "score")
                self.assertEqual(score_step.status, "succeeded")
                self.assertEqual(score_step.reason, "scored_fallback")

                self.assertEqual(len(store.records), 1)
                stored_rec = list(store.records.values())[0]
                self.assertTrue(stored_rec.score.reasoning.startswith(FALLBACK_REASONING_PREFIX))
                self.assertEqual(result.category, stored_rec.score.category)
                self.assertEqual(result.score, stored_rec.score.score)

    def test_allow_fallback_true_model_output_json_decode_and_validation_error(self):
        """JSONDecodeError and ValidationError (both ValueError subclasses) trigger fallback."""
        try:
            json.loads("{invalid json")
        except json.JSONDecodeError as e:
            json_err = e

        try:
            LeadScore(score=500, reasoning="r", suggested_action="a", priority_level="low")
        except ValidationError as e:
            validation_err = e

        for exc in (json_err, validation_err):
            with self.subTest(exc=type(exc).__name__):
                store = FakeStore()
                deps = make_deps(
                    store=store,
                    score_lead=MagicMock(side_effect=exc),
                )
                lead = make_sample_lead()
                result = run_pipeline(lead, deps=deps, allow_fallback_score=True)

                self.assertEqual(result.status, "completed")
                score_step = next(s for s in result.steps if s.name == "score")
                self.assertEqual(score_step.status, "succeeded")
                self.assertEqual(score_step.reason, "scored_fallback")
                self.assertEqual(len(store.records), 1)

    def test_allow_fallback_true_disallowed_categories_do_not_fallback(self):
        """Config and unexpected errors (RuntimeError, AuthenticationError, Exception) never fall back."""
        disallowed = [
            RuntimeError("OPENROUTER_API_KEY missing"),
            make_authentication_error("invalid api key"),
            Exception("boom"),
        ]
        for exc in disallowed:
            with self.subTest(exc=type(exc).__name__):
                fallback_spy = MagicMock()
                call_log = []
                store = FakeStore(call_log=call_log)
                deps = make_deps(
                    call_log=call_log,
                    store=store,
                    score_lead=MagicMock(side_effect=exc),
                    fallback_score=fallback_spy,
                )
                lead = make_sample_lead()
                result = run_pipeline(lead, deps=deps, allow_fallback_score=True)

                fallback_spy.assert_not_called()
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.failed_step, "score")
                self.assertFalse(result.persisted)
                self.assertNotIn("store", call_log)
                self.assertEqual(len(store.records), 0)

                # Also test with default fallback_score (not mocked) to ensure real fallback is not executed
                store_default = FakeStore()
                deps_default = make_deps(
                    store=store_default,
                    score_lead=MagicMock(side_effect=exc),
                )
                res_def = run_pipeline(lead, deps=deps_default, allow_fallback_score=True)
                self.assertEqual(res_def.status, "failed")
                self.assertEqual(res_def.failed_step, "score")
                self.assertFalse(res_def.persisted)
                self.assertEqual(len(store_default.records), 0)

    def test_allow_fallback_true_when_ai_score_succeeds_fallback_never_called(self):
        """When AI scoring succeeds, score step has reason 'scored' and fallback is not called."""
        fallback_spy = MagicMock()
        deps = make_deps(fallback_score=fallback_spy)
        lead = make_sample_lead()
        result = run_pipeline(lead, deps=deps, allow_fallback_score=True)

        fallback_spy.assert_not_called()
        self.assertEqual(result.status, "completed")
        score_step = next(s for s in result.steps if s.name == "score")
        self.assertEqual(score_step.status, "succeeded")
        self.assertEqual(score_step.reason, "scored")

    def test_injected_fallback_scorer_used_when_present_and_default_when_none(self):
        """deps.fallback_score is called when provided; otherwise default fallback_score is called."""
        custom_score = LeadScore(
            score=45,
            reasoning=FALLBACK_REASONING_PREFIX + "custom injected",
            suggested_action="Contact soon.",
            priority_level="normal",
        )
        custom_mock = MagicMock(return_value=custom_score)
        store = FakeStore()
        deps_custom = make_deps(
            store=store,
            score_lead=MagicMock(side_effect=make_rate_limit_error()),
            fallback_score=custom_mock,
        )
        lead = make_sample_lead()
        res_custom = run_pipeline(lead, deps=deps_custom, allow_fallback_score=True)

        custom_mock.assert_called_once()
        self.assertEqual(res_custom.score, 45)

        # Now test when fallback_score is None (uses default)
        store_default = FakeStore()
        deps_default = make_deps(
            store=store_default,
            score_lead=MagicMock(side_effect=make_rate_limit_error()),
            fallback_score=None,
        )
        res_default = run_pipeline(lead, deps=deps_default, allow_fallback_score=True)
        self.assertEqual(res_default.status, "completed")
        stored_default = list(store_default.records.values())[0]
        self.assertTrue(stored_default.score.reasoning.startswith(FALLBACK_REASONING_PREFIX))

    def test_fallback_scorer_raising_takes_original_failure_path(self):
        """If the fallback scorer itself raises, pipeline fails with the original error classification."""
        def failing_fallback(lead):
            raise RuntimeError("fallback internal crash")

        call_log = []
        store = FakeStore(call_log=call_log)
        deps = make_deps(
            call_log=call_log,
            store=store,
            score_lead=MagicMock(side_effect=make_rate_limit_error("rate limit original")),
            fallback_score=failing_fallback,
        )
        lead = make_sample_lead()
        result = run_pipeline(lead, deps=deps, allow_fallback_score=True)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failed_step, "score")
        self.assertFalse(result.persisted)
        self.assertIn("score", result.retry_steps)
        self.assertTrue(result.to_dict()["retryable"])
        score_step = next(s for s in result.steps if s.name == "score")
        self.assertEqual(score_step.error.category, "transient")
        self.assertNotIn("store", call_log)

    def test_first_write_wins_authoritative_stored_record(self):
        """First write wins: second run with fallback retains original stored score and category."""
        store = FakeStore()
        lead_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        lead = make_sample_lead()

        # Run 1: Normal AI score 87
        deps1 = make_deps(store=store)
        res1 = run_pipeline(lead, lead_id=lead_id, deps=deps1, allow_fallback_score=False)
        self.assertEqual(res1.status, "completed")
        self.assertEqual(res1.created, True)
        self.assertEqual(res1.score, 87)
        self.assertEqual(res1.category, "hot")

        # Run 2: AI fails transiently, allow_fallback_score=True
        deps2 = make_deps(
            store=store,
            score_lead=MagicMock(side_effect=make_rate_limit_error()),
        )
        res2 = run_pipeline(lead, lead_id=lead_id, deps=deps2, allow_fallback_score=True)
        self.assertEqual(res2.created, False)
        self.assertEqual(res2.score, 87)
        self.assertEqual(res2.category, "hot")
        score_step = next(s for s in res2.steps if s.name == "score")
        self.assertEqual(score_step.reason, "scored_fallback")

        for outbound in ("email", "sms", "crm"):
            step = next(s for s in res2.steps if s.name == outbound)
            self.assertEqual(step.status, "skipped")
            self.assertEqual(step.reason, "already_done")

    def test_sparse_lead_scored_cold_skips_outbound_steps(self):
        """A sparse lead scored cold by fallback skips email, sms, and crm with 'cold_lead'."""
        store = FakeStore()
        deps = make_deps(
            store=store,
            score_lead=MagicMock(side_effect=make_rate_limit_error()),
        )
        sparse_lead = LeadInput(
            name="Sparse Lead",
            email="sparse@example.com",
            phone="+14155550192",
            message="Hello",
        )
        result = run_pipeline(sparse_lead, deps=deps, allow_fallback_score=True)

        self.assertEqual(result.status, "completed")
        score_step = next(s for s in result.steps if s.name == "score")
        self.assertEqual(score_step.reason, "scored_fallback")
        self.assertEqual(result.category, "cold")

        for outbound in ("email", "sms", "crm"):
            step = next(s for s in result.steps if s.name == outbound)
            self.assertEqual(step.status, "skipped")
            self.assertEqual(step.reason, "cold_lead")

    def test_logging_and_privacy_canary_protection(self):
        """Emits exactly one warning on fallback; logs and serialized result leak no customer PII."""
        store = FakeStore()
        deps = make_deps(
            store=store,
            score_lead=MagicMock(side_effect=make_rate_limit_error("OpenAI rate limit")),
        )
        canary_name = "Zed Canary"
        canary_email = "canary.pipeline@example.com"
        canary_phone = "+14155550149"
        canary_msg = "Super secret canary message content"

        canary_lead = LeadInput(
            name=canary_name,
            email=canary_email,
            phone=canary_phone,
            message=canary_msg,
        )

        with self.assertLogs("services.pipeline_service", level="WARNING") as log_cm:
            result = run_pipeline(canary_lead, deps=deps, allow_fallback_score=True)

        self.assertEqual(len(log_cm.records), 1)
        record = log_cm.records[0]
        self.assertEqual(record.levelno, logging.WARNING)
        self.assertIn("pipeline score fell back to rules", record.getMessage())
        self.assertIn(result.lead_id, record.getMessage())
        self.assertIn("transient", record.getMessage())
        self.assertIn("RateLimitError", record.getMessage())

        # Check logs for leaks
        for log_text in log_cm.output:
            for canary in (canary_name, canary_email, canary_phone, canary_msg):
                self.assertNotIn(canary, log_text)

        # Check serialized result for leaks
        result_json = json.dumps(result.to_dict())
        for canary in (canary_name, canary_email, canary_phone, canary_msg):
            self.assertNotIn(canary, result_json)

    def test_pipeline_result_to_dict_has_same_key_set_for_fallback(self):
        """PipelineResult.to_dict() preserves the exact key set on a fallback run."""
        deps_normal = make_deps()
        lead = make_sample_lead()
        res_normal = run_pipeline(lead, deps=deps_normal)
        normal_keys = set(res_normal.to_dict().keys())

        deps_fallback = make_deps(score_lead=MagicMock(side_effect=make_rate_limit_error()))
        res_fallback = run_pipeline(lead, deps=deps_fallback, allow_fallback_score=True)
        fallback_keys = set(res_fallback.to_dict().keys())

        self.assertEqual(normal_keys, fallback_keys)

    def test_pipeline_service_source_hygiene(self):
        """pipeline_service.py must be pure ASCII and contain no os.getenv or os.environ."""
        source_path = pipeline_service.__file__
        with open(source_path, "rb") as f:
            raw = f.read()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"pipeline_service.py is not pure ASCII: {exc}")

        content = raw.decode("ascii")
        self.assertNotIn("os.getenv", content)
        self.assertNotIn("os.environ", content)


if __name__ == "__main__":
    unittest.main()
