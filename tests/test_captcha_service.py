"""
tests/test_captcha_service.py

Unit tests for services/captcha_service.py.

Uses only the standard library and httpx.MockTransport.
No real network calls are made and no .env files are read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from httpx import Request

import services.captcha_service as cs
from services.captcha_service import CaptchaError, verify_token


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _mock_client(handler) -> httpx.Client:
    """Return a fresh httpx.Client wired to an httpx.MockTransport."""
    return httpx.Client(transport=httpx.MockTransport(handler))


# ===========================================================================
# Group A: Import & Source Hygiene
# ===========================================================================

class TestGroupAImportHygiene(unittest.TestCase):
    """Verify module imports cleanly with empty env and source is clean."""

    def test_a01_import_empty_environment_does_not_raise(self):
        cmd = [
            sys.executable,
            "-c",
            "import os, services.captcha_service; assert hasattr(services.captcha_service, 'verify_token')",
        ]
        res = subprocess.run(
            cmd,
            env={"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Import in empty env failed: {res.stderr}")

    def test_a02_source_is_pure_ascii(self):
        source_path = Path(cs.__file__)
        raw_bytes = source_path.read_bytes()
        try:
            raw_bytes.decode("ascii")
        except UnicodeDecodeError as err:
            self.fail(f"Source file {source_path} is not pure ASCII: {err}")
        self.assertTrue(all(b < 128 for b in raw_bytes))

    def test_a03_source_contains_no_logging_or_print(self):
        source_path = Path(cs.__file__)
        text = source_path.read_text(encoding="ascii")
        self.assertNotIn("import logging", text)
        self.assertNotIn("print(", text)


# ===========================================================================
# Group B: Secret Key Enforcement
# ===========================================================================

class TestGroupBSecretKey(unittest.TestCase):
    """Verify TURNSTILE_SECRET_KEY handling and call-time reading."""

    def test_b01_secret_missing_empty_whitespace_raises_config(self):
        cases = [
            ("missing", None),
            ("empty", ""),
            ("whitespace", "   "),
        ]
        for label, val in cases:
            with self.subTest(label=label):
                env = {} if val is None else {"TURNSTILE_SECRET_KEY": val}
                with patch.dict(os.environ, env, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=AssertionError("HTTP client must not be called"),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("test-token")
                        self.assertEqual(ctx.exception.category, "config")
                        self.assertIn("TURNSTILE_SECRET_KEY", str(ctx.exception))

    def test_b02_secret_read_at_call_time(self):
        captured_secrets: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            parsed = urllib.parse.parse_qs(request.read().decode("utf-8"))
            captured_secrets.append(parsed.get("secret", [""])[0])
            return httpx.Response(200, json={"success": True})

        with patch(
            "services.captcha_service._get_http_client",
            side_effect=lambda: _mock_client(handler),
        ):
            with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "secret-alpha"}, clear=True):
                res1 = verify_token("tok-1")
                self.assertTrue(res1["verified"])

            with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "secret-beta"}, clear=True):
                res2 = verify_token("tok-2")
                self.assertTrue(res2["verified"])

        self.assertEqual(captured_secrets, ["secret-alpha", "secret-beta"])


# ===========================================================================
# Group C: Token Handling
# ===========================================================================

class TestGroupCTokenHandling(unittest.TestCase):
    """Verify token type checks, lengths, and whitespace trimming."""

    def test_c01_missing_or_blank_token_returns_missing_token_no_network(self):
        tokens = [None, "", "   ", 123, [], {}]
        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "secret"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=AssertionError("HTTP client must not be called"),
            ):
                for tok in tokens:
                    with self.subTest(token=tok):
                        res = verify_token(tok)
                        self.assertEqual(
                            res,
                            {"verified": False, "reason": "missing_token", "error_codes": []},
                        )

    def test_c02_token_length_boundaries(self):
        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "secret"}, clear=True):
            # 2049 characters -> invalid_token, no network
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=AssertionError("HTTP client must not be called"),
            ):
                res_oversize = verify_token("x" * 2049)
                self.assertEqual(
                    res_oversize,
                    {"verified": False, "reason": "invalid_token", "error_codes": []},
                )

            # 2048 characters -> reaches network
            called: list[Request] = []

            def handler(request: httpx.Request) -> httpx.Response:
                called.append(request)
                return httpx.Response(200, json={"success": True})

            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(handler),
            ):
                res_valid = verify_token("x" * 2048)
                self.assertEqual(
                    res_valid,
                    {"verified": True, "reason": "ok", "error_codes": []},
                )
                self.assertEqual(len(called), 1)

    def test_c03_surrounding_whitespace_stripped_before_sending(self):
        captured_body: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_body.append(request.read().decode("utf-8"))
            return httpx.Response(200, json={"success": True})

        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "secret"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(handler),
            ):
                verify_token("   padded-token-value   ")

        self.assertEqual(len(captured_body), 1)
        parsed = urllib.parse.parse_qs(captured_body[0])
        self.assertEqual(parsed.get("response"), ["padded-token-value"])


# ===========================================================================
# Group D: Request Shape & Remote IP
# ===========================================================================

class TestGroupDRequestShape(unittest.TestCase):
    """Verify POST method, URL, headers, and remote IP handling."""

    def test_d01_request_shape_and_url(self):
        captured_req: list[Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_req.append(request)
            return httpx.Response(200, json={"success": True})

        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "test-key"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(handler),
            ):
                verify_token("test-tok", remote_ip="203.0.113.9")

        self.assertEqual(len(captured_req), 1)
        req = captured_req[0]
        self.assertEqual(req.method, "POST")
        self.assertEqual(
            str(req.url), "https://challenges.cloudflare.com/turnstile/v0/siteverify"
        )
        self.assertTrue(
            req.headers.get("content-type", "").startswith("application/x-www-form-urlencoded")
        )
        parsed = urllib.parse.parse_qs(req.read().decode("utf-8"))
        self.assertEqual(parsed.get("secret"), ["test-key"])
        self.assertEqual(parsed.get("response"), ["test-tok"])
        self.assertEqual(parsed.get("remoteip"), ["203.0.113.9"])

    def test_d02_remote_ip_presence_and_absence(self):
        cases = [
            ("ipv4", "203.0.113.9", "203.0.113.9"),
            ("ipv6", "2001:db8::1", "2001:db8::1"),
            ("none", None, None),
            ("int", 123, None),
            ("invalid_text", "not-an-ip", None),
            ("invalid_octets", "999.1.1.1", None),
            ("empty_str", "", None),
        ]

        for label, ip_val, expected_in_req in cases:
            with self.subTest(label=label):
                captured: list[Request] = []

                def handler(request: httpx.Request) -> httpx.Response:
                    captured.append(request)
                    return httpx.Response(200, json={"success": True})

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        verify_token("tok", remote_ip=ip_val)

                parsed = urllib.parse.parse_qs(captured[0].read().decode("utf-8"))
                if expected_in_req is None:
                    self.assertNotIn("remoteip", parsed)
                else:
                    self.assertEqual(parsed.get("remoteip"), [expected_in_req])

    def test_d03_token_special_characters_round_trip(self):
        special_token = "&=+ %"
        captured: list[Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"success": True})

        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(handler),
            ):
                verify_token(special_token)

        parsed = urllib.parse.parse_qs(captured[0].read().decode("utf-8"))
        self.assertEqual(parsed.get("response"), [special_token])


# ===========================================================================
# Group E: Success & Hostname Enforcement
# ===========================================================================

class TestGroupESuccessAndHostname(unittest.TestCase):
    """Verify success boolean evaluation and hostname check."""

    def test_e01_success_true_vs_falsy_or_non_boolean(self):
        # success is True -> ok
        def ok_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"success": True})

        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(ok_handler),
            ):
                res = verify_token("tok")
                self.assertEqual(
                    res, {"verified": True, "reason": "ok", "error_codes": []}
                )

        # non-boolean success values -> rejected
        non_booleans = [
            ("str_true", {"success": "true"}),
            ("int_one", {"success": 1}),
            ("none", {"success": None}),
            ("missing", {}),
        ]
        for label, body in non_booleans:
            with self.subTest(label=label):
                def handler(request: httpx.Request, b=body) -> httpx.Response:
                    return httpx.Response(200, json=b)

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        res = verify_token("tok")
                        self.assertEqual(
                            res, {"verified": False, "reason": "rejected", "error_codes": []}
                        )

    def test_e02_hostname_enforcement(self):
        cases = [
            ("exact_match", "example.com", True, "ok"),
            ("uppercase_match", "EXAMPLE.COM", True, "ok"),
            ("padded_match", "  example.com  ", True, "ok"),
            ("mismatch", "attacker.com", False, "hostname_mismatch"),
            ("missing_hostname", None, False, "hostname_mismatch"),
        ]

        env_with_host = {
            "TURNSTILE_SECRET_KEY": "k",
            "TURNSTILE_EXPECTED_HOSTNAME": "example.com",
        }

        for label, host_val, expected_verified, expected_reason in cases:
            with self.subTest(label=label):
                body = {"success": True}
                if host_val is not None:
                    body["hostname"] = host_val

                def handler(request: httpx.Request, b=body) -> httpx.Response:
                    return httpx.Response(200, json=b)

                with patch.dict(os.environ, env_with_host, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        res = verify_token("tok")
                        self.assertEqual(res["verified"], expected_verified)
                        self.assertEqual(res["reason"], expected_reason)
                        self.assertEqual(res["error_codes"], [])

        # Unset expected hostname -> any hostname passes
        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
            for unset_case in ["any.domain.org", None]:
                with self.subTest(unset_case=unset_case):
                    body = {"success": True}
                    if unset_case is not None:
                        body["hostname"] = unset_case

                    def handler(request: httpx.Request, b=body) -> httpx.Response:
                        return httpx.Response(200, json=b)

                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        res = verify_token("tok")
                        self.assertEqual(
                            res, {"verified": True, "reason": "ok", "error_codes": []}
                        )


# ===========================================================================
# Group F: Error-Code Table & Precedence
# ===========================================================================

class TestGroupFErrorCodes(unittest.TestCase):
    """Verify error-code mapping, precedence, and sanitisation."""

    def test_f01_error_code_reasons(self):
        cases = [
            ("expired_or_used", ["timeout-or-duplicate"], "expired_or_used"),
            ("missing_response", ["missing-input-response"], "invalid_token"),
            ("invalid_response", ["invalid-input-response"], "invalid_token"),
            ("bad_request", ["bad-request"], "rejected"),
            ("unknown_code", ["custom-unknown-error"], "rejected"),
            ("empty_list", [], "rejected"),
            ("missing_codes_key", None, "rejected"),
        ]

        for label, codes_val, expected_reason in cases:
            with self.subTest(label=label):
                body: dict[str, object] = {"success": False}
                if codes_val is not None:
                    body["error-codes"] = codes_val

                def handler(request: httpx.Request, b=body) -> httpx.Response:
                    return httpx.Response(200, json=b)

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        res = verify_token("tok")
                        self.assertFalse(res["verified"])
                        self.assertEqual(res["reason"], expected_reason)
                        self.assertEqual(
                            res["error_codes"],
                            codes_val if codes_val is not None else [],
                        )

    def test_f02_secret_and_internal_errors(self):
        error_cases = [
            ("missing_secret", ["missing-input-secret"], "config"),
            ("invalid_secret", ["invalid-input-secret"], "config"),
            ("internal_error", ["internal-error"], "transient"),
        ]

        for label, codes, expected_category in error_cases:
            with self.subTest(label=label):
                def handler(request: httpx.Request, c=codes) -> httpx.Response:
                    return httpx.Response(200, json={"success": False, "error-codes": c})

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("tok")
                        self.assertEqual(ctx.exception.category, expected_category)

    def test_f03_precedence(self):
        cases = [
            (
                "secret_over_response",
                ["invalid-input-response", "invalid-input-secret"],
                "config",
            ),
            (
                "internal_over_response",
                ["invalid-input-response", "internal-error"],
                "transient",
            ),
            (
                "secret_over_internal",
                ["internal-error", "invalid-input-secret"],
                "config",
            ),
        ]

        for label, codes, expected_cat in cases:
            with self.subTest(label=label):
                def handler(request: httpx.Request, c=codes) -> httpx.Response:
                    return httpx.Response(200, json={"success": False, "error-codes": c})

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("tok")
                        self.assertEqual(ctx.exception.category, expected_cat)

    def test_f04_error_code_sanitising(self):
        raw_codes = [
            "  with spaces  ",
            "quote'and\"symbols",
            123,
            None,
            "x" * 80,
            "",
            "normal-code_1",
        ]
        body = {"success": False, "error-codes": raw_codes}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(handler),
            ):
                res = verify_token("tok")

        expected_clean = [
            "withspaces",
            "quoteandsymbols",
            "x" * 64,
            "normal-code_1",
        ]
        self.assertEqual(res["error_codes"], expected_clean)

        # Truncation to max 10 codes
        many_codes = [f"code-{i}" for i in range(20)]

        def handler_many(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"success": False, "error-codes": many_codes})

        with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
            with patch(
                "services.captcha_service._get_http_client",
                side_effect=lambda: _mock_client(handler_many),
            ):
                res_many = verify_token("tok")
        self.assertEqual(len(res_many["error_codes"]), 10)

        # Non-list error-codes
        for non_list in ["string", 42, {"a": "b"}]:
            with self.subTest(non_list=non_list):
                def handler_nl(request: httpx.Request, nl=non_list) -> httpx.Response:
                    return httpx.Response(200, json={"success": False, "error-codes": nl})

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler_nl),
                    ):
                        res_nl = verify_token("tok")
                        self.assertEqual(res_nl["error_codes"], [])


# ===========================================================================
# Group G: Transport & HTTP Status
# ===========================================================================

class TestGroupGTransportAndStatus(unittest.TestCase):
    """Verify handling of transport errors, status codes, and unreadable responses."""

    def test_g01_network_timeout_and_connect_error(self):
        exceptions = [
            httpx.ConnectTimeout("connection timed out"),
            httpx.ConnectError("failed to connect"),
            httpx.ReadTimeout("read timed out"),
        ]

        for exc in exceptions:
            with self.subTest(exc=type(exc).__name__):
                def handler(request: httpx.Request, e=exc) -> httpx.Response:
                    raise e

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("tok")
                        self.assertEqual(ctx.exception.category, "transient")
                        self.assertIn("Could not reach", str(ctx.exception))

    def test_g02_transient_http_statuses(self):
        for status in [429, 500, 502, 503, 504]:
            with self.subTest(status=status):
                def handler(request: httpx.Request, s=status) -> httpx.Response:
                    return httpx.Response(s, text="error")

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("tok")
                        self.assertEqual(ctx.exception.category, "transient")
                        self.assertIn("temporarily unavailable", str(ctx.exception))

    def test_g03_config_http_statuses(self):
        for status in [400, 401, 403, 404]:
            with self.subTest(status=status):
                def handler(request: httpx.Request, s=status) -> httpx.Response:
                    return httpx.Response(s, text="rejected")

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("tok")
                        self.assertEqual(ctx.exception.category, "config")
                        self.assertIn(f"HTTP {status}", str(ctx.exception))

    def test_g04_unreadable_json_bodies(self):
        unreadable_responses = [
            ("invalid_json", httpx.Response(200, text="not json content")),
            ("json_list", httpx.Response(200, json=["item1", "item2"])),
            ("json_string", httpx.Response(200, json="simple string")),
            ("json_int", httpx.Response(200, json=123)),
        ]

        for label, resp in unreadable_responses:
            with self.subTest(label=label):
                def handler(request: httpx.Request, r=resp) -> httpx.Response:
                    return r

                with patch.dict(os.environ, {"TURNSTILE_SECRET_KEY": "k"}, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        with self.assertRaises(CaptchaError) as ctx:
                            verify_token("tok")
                        self.assertEqual(ctx.exception.category, "transient")
                        self.assertIn("unreadable response", str(ctx.exception))


# ===========================================================================
# Group H: Privacy & Canary Leak Prevention
# ===========================================================================

class TestGroupHPrivacyCanaries(unittest.TestCase):
    """Verify that canaries never leak into exceptions or returned dictionaries."""

    CANARY_TOKEN = "CANARY-TOKEN-xyz"
    CANARY_SECRET = "CANARY-SECRET-xyz"
    CANARY_IP = "198.51.100.77"
    CANARY_BODY = "CANARY-BODY-xyz"

    def _assert_no_canary(self, obj: object):
        text = json.dumps(obj) if isinstance(obj, dict) else str(obj)
        self.assertNotIn(self.CANARY_TOKEN, text)
        self.assertNotIn(self.CANARY_SECRET, text)
        self.assertNotIn(self.CANARY_IP, text)
        self.assertNotIn(self.CANARY_BODY, text)

    def test_h01_canary_leak_checks_across_all_failure_modes(self):
        modes = [
            # Mode 1: Secret error in config
            ("secret_missing", lambda: httpx.Response(200, json={"success": True}), None),
            # Mode 2: Hostname mismatch
            (
                "hostname_mismatch",
                lambda: httpx.Response(
                    200,
                    json={
                        "success": True,
                        "hostname": "attacker.com",
                        "message": self.CANARY_BODY,
                    },
                ),
                "example.com",
            ),
            # Mode 3: Provider secret rejection
            (
                "provider_secret_rejection",
                lambda: httpx.Response(
                    200,
                    json={
                        "success": False,
                        "error-codes": ["invalid-input-secret"],
                        "message": self.CANARY_BODY,
                    },
                ),
                None,
            ),
            # Mode 4: Provider internal error
            (
                "provider_internal_error",
                lambda: httpx.Response(
                    200,
                    json={
                        "success": False,
                        "error-codes": ["internal-error"],
                        "message": self.CANARY_BODY,
                    },
                ),
                None,
            ),
            # Mode 5: Rejected token with message body
            (
                "rejected_token",
                lambda: httpx.Response(
                    200,
                    json={
                        "success": False,
                        "error-codes": ["bad-request"],
                        "message": self.CANARY_BODY,
                    },
                ),
                None,
            ),
            # Mode 6: HTTP 500 error with message
            (
                "http_500",
                lambda: httpx.Response(500, text=f"error: {self.CANARY_BODY}"),
                None,
            ),
            # Mode 7: HTTP 403 error with message
            (
                "http_403",
                lambda: httpx.Response(403, text=f"forbidden: {self.CANARY_BODY}"),
                None,
            ),
            # Mode 8: Unreadable JSON body containing canary
            (
                "unreadable_json",
                lambda: httpx.Response(200, text=f"not json {self.CANARY_BODY}"),
                None,
            ),
            # Mode 9: Network timeout
            (
                "network_timeout",
                lambda: (_ for _ in ()).throw(httpx.ConnectTimeout(self.CANARY_BODY)),
                None,
            ),
        ]

        for label, resp_factory, expected_host in modes:
            with self.subTest(label=label):
                def handler(request: httpx.Request, rf=resp_factory) -> httpx.Response:
                    return rf()

                env = {
                    "TURNSTILE_SECRET_KEY": self.CANARY_SECRET,
                }
                if expected_host is not None:
                    env["TURNSTILE_EXPECTED_HOSTNAME"] = expected_host

                if label == "secret_missing":
                    env = {}

                with patch.dict(os.environ, env, clear=True):
                    with patch(
                        "services.captcha_service._get_http_client",
                        side_effect=lambda: _mock_client(handler),
                    ):
                        try:
                            res = verify_token(
                                self.CANARY_TOKEN, remote_ip=self.CANARY_IP
                            )
                            self._assert_no_canary(res)
                        except Exception as exc:
                            self._assert_no_canary(exc)


if __name__ == "__main__":
    unittest.main()
