"""
tests/test_llm_output_shapes.py

Unit tests for LLM output shape validation and defensive parsing in services/llm_service.py.
Verifies that malformed or non-dict model replies raise clean ValueError exceptions
without leaking model content, and that valid fenced and bare JSON replies continue to parse.

Follows project testing standards:
- Pure ASCII source only, LF line endings, no BOM.
- unittest only, no network calls.
"""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from models import EmailPayload, LeadInput, LeadScore, SMSPayload
from services.llm_service import _strip_fences, generate_email, generate_sms, score_lead


class FakeClient:
    """Mock OpenAI client that returns a pre-configured response object."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.kwargs = kwargs
                return self._outer.response

        class _Chat:
            def __init__(self, outer):
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


def _make_sample_lead() -> LeadInput:
    """Sample LeadInput for testing."""
    return LeadInput(
        name="Sarah Mitchell",
        email="sarah.mitchell@example.com",
        phone="+14155550192",
        message="Looking for a 3-bedroom house in Austin.",
        property_type="house",
        preferred_location="Austin, TX",
        budget="$450k-$550k",
        timeline="within 3 months",
    )


def _make_sample_score() -> LeadScore:
    """Sample LeadScore for email and SMS generators."""
    return LeadScore(
        score=85,
        category="hot",
        reasoning="Pre-approved buyer looking to purchase within 3 months in Austin.",
        suggested_action="Schedule an in-person viewing immediately.",
        priority_level="urgent",
    )


class TestLLMOutputShapes(unittest.TestCase):
    """Test suite for malformed and edge-case LLM output shapes."""

    def setUp(self):
        self.lead = _make_sample_lead()
        self.score = _make_sample_score()

    def _call_function(self, fn_name: str):
        """Invoke the target function with the standard test fixtures."""
        if fn_name == "score_lead":
            return score_lead(self.lead)
        if fn_name == "generate_email":
            return generate_email(self.lead, self.score)
        if fn_name == "generate_sms":
            return generate_sms(self.lead, self.score)
        raise ValueError(f"Unknown function name: {fn_name}")

    def test_strip_fences_table(self):
        """Verify _strip_fences handles all valid, inline, unclosed, empty, and non-fenced inputs."""
        F = "`" * 3
        cases = [
            (F + '{"a":1}' + F, '{"a":1}'),
            (F + 'json{"a":1}' + F, '{"a":1}'),
            (F + '{"a":1}\n' + F, '{"a":1}'),
            (F + 'json\n{"a":1}\n' + F, '{"a":1}'),
            (F + 'json\n{"a":1}\n' + F + '\n', '{"a":1}'),
            (F + '{"a":1}', '{"a":1}'),
            ('{"a":1}', '{"a":1}'),
            (F, ''),
            (F + 'json', ''),
            (F + 'json\n', ''),
            (F + F, ''),
            (F + '\n', ''),
            ('plain_word', 'plain_word'),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw, expected=expected):
                self.assertEqual(_strip_fences(raw), expected)

    def test_malformed_replies_raise_value_error_for_all_functions(self):
        """Verify that all malformed reply shapes raise ValueError and no other exception type."""
        cases = [
            ("choices_empty_list", SimpleNamespace(choices=[])),
            ("choices_none", SimpleNamespace(choices=None)),
            ("no_choices_attribute", SimpleNamespace()),
            (
                "content_none",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
                ),
            ),
            (
                "content_int",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=123))]
                ),
            ),
            (
                "content_empty_str",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
                ),
            ),
            (
                "content_whitespace",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
                ),
            ),
            (
                "content_lone_fence",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="```"))]
                ),
            ),
            (
                "content_fence_json_tag_only",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="```json"))]
                ),
            ),
            (
                "content_empty_fenced_block",
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="```json\n```"))
                    ]
                ),
            ),
            (
                "content_not_json",
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="not json"))
                    ]
                ),
            ),
            (
                "content_json_list",
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="[1,2,3]"))
                    ]
                ),
            ),
            (
                "content_json_string",
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content='"a string"'))
                    ]
                ),
            ),
            (
                "content_json_number",
                SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="42"))]
                ),
            ),
        ]

        functions = ["score_lead", "generate_email", "generate_sms"]

        for fn_name in functions:
            for case_name, response in cases:
                with self.subTest(function=fn_name, case=case_name):
                    with patch(
                        "services.llm_service._get_client",
                        return_value=FakeClient(response),
                    ):
                        with self.assertRaises(ValueError) as ctx:
                            self._call_function(fn_name)
                        # Ensure the exception is exactly ValueError or a direct subclass (like JSONDecodeError)
                        self.assertIsInstance(ctx.exception, ValueError)
                        # Explicitly ensure unexpected error types are not raised
                        self.assertNotIsInstance(ctx.exception, IndexError)
                        self.assertNotIsInstance(ctx.exception, AttributeError)
                        self.assertNotIsInstance(ctx.exception, TypeError)

    def test_unindexable_choices_raise_value_error_without_cause(self):
        """choices={"a": 1} and choices=object() raise ValueError with no chained cause and suppressed context."""
        fixed_msg = "The model returned an empty or malformed reply."
        unindexable_cases = [
            ("choices_dict", SimpleNamespace(choices={"a": 1})),
            ("choices_object_no_len", SimpleNamespace(choices=object())),
        ]
        functions = ["score_lead", "generate_email", "generate_sms"]

        for fn_name in functions:
            for case_name, response in unindexable_cases:
                with self.subTest(function=fn_name, case=case_name):
                    with patch(
                        "services.llm_service._get_client",
                        return_value=FakeClient(response),
                    ):
                        with self.assertRaises(ValueError) as ctx:
                            self._call_function(fn_name)
                        exc = ctx.exception
                        self.assertEqual(str(exc), fixed_msg)
                        self.assertIsNone(exc.__cause__)
                        self.assertTrue(exc.__suppress_context__)

    def test_canary_leak_protection(self):
        """Assert that model reply content containing a canary is never leaked in the exception message."""
        canary = "CANARY-REPLY-xyz"
        canary_responses = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f"Error {canary} occurred")
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=f"```json\n{canary}\n```"
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps([1, 2, canary])
                        )
                    )
                ]
            ),
        ]

        functions = ["score_lead", "generate_email", "generate_sms"]

        for fn_name in functions:
            for idx, response in enumerate(canary_responses):
                with self.subTest(function=fn_name, canary_case=idx):
                    with patch(
                        "services.llm_service._get_client",
                        return_value=FakeClient(response),
                    ):
                        with self.assertRaises(ValueError) as ctx:
                            self._call_function(fn_name)
                        err_msg = str(ctx.exception)
                        self.assertNotIn(canary, err_msg)

    def test_regression_valid_fenced_and_bare_replies(self):
        """Regression: valid bare JSON and normally/inline fenced JSON return identical valid models."""
        F = "`" * 3
        score_dict = {
            "score": 88,
            "category": "hot",
            "reasoning": "High budget buyer ready to purchase immediately.",
            "suggested_action": "Contact lead directly via phone.",
            "priority_level": "urgent",
        }
        score_json = json.dumps(score_dict)

        email_dict = {
            "subject": "Homes in Austin that match your search",
            "body": "Hi Sarah,\n\nI would be delighted to assist with your search for a home in Austin.",
        }
        email_json = json.dumps(email_dict)

        sms_dict = {
            "message": "Hi Sarah, thanks for your inquiry about Austin homes! An agent will call you shortly.",
        }
        sms_json = json.dumps(sms_dict)

        # 1. score_lead regression
        score_variants = [
            ("bare", score_json),
            ("standard_fenced", f"{F}json\n{score_json}\n{F}"),
            ("inline_fenced", f"{F}json{score_json}{F}"),
            ("no_tag_trailing_nl", f"{F}{score_json}\n{F}"),
        ]
        for var_name, content in score_variants:
            resp = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
            with self.subTest(step="score_lead", variant=var_name):
                with patch(
                    "services.llm_service._get_client",
                    return_value=FakeClient(resp),
                ):
                    res = score_lead(self.lead)
                    self.assertIsInstance(res, LeadScore)
                    self.assertEqual(res.score, 88)
                    self.assertEqual(res.category, "hot")
                    self.assertEqual(res.priority_level, "urgent")

        # 2. generate_email regression
        email_variants = [
            ("bare", email_json),
            ("standard_fenced", f"{F}json\n{email_json}\n{F}"),
            ("inline_fenced", f"{F}json{email_json}{F}"),
            ("no_tag_trailing_nl", f"{F}{email_json}\n{F}"),
        ]
        for var_name, content in email_variants:
            resp = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
            with self.subTest(step="generate_email", variant=var_name):
                with patch(
                    "services.llm_service._get_client",
                    return_value=FakeClient(resp),
                ):
                    res = generate_email(self.lead, self.score)
                    self.assertIsInstance(res, EmailPayload)
                    self.assertEqual(res.subject, email_dict["subject"])
                    self.assertIn("Hi Sarah", res.body)

        # 3. generate_sms regression
        sms_variants = [
            ("bare", sms_json),
            ("standard_fenced", f"{F}json\n{sms_json}\n{F}"),
            ("inline_fenced", f"{F}json{sms_json}{F}"),
            ("no_tag_trailing_nl", f"{F}{sms_json}\n{F}"),
        ]
        for var_name, content in sms_variants:
            resp = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
            with self.subTest(step="generate_sms", variant=var_name):
                with patch(
                    "services.llm_service._get_client",
                    return_value=FakeClient(resp),
                ):
                    res = generate_sms(self.lead, self.score)
                    self.assertIsInstance(res, SMSPayload)
                    self.assertIn("Hi Sarah", res.message)


if __name__ == "__main__":
    unittest.main()
