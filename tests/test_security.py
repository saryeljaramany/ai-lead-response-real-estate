"""
tests/test_security.py

Unit and integration tests for services/security.py.
Enforces import safety, constant-time comparisons, API key dependency rules,
fail-closed behaviors, throwaway app route protection, and OAuth state token
generation, verification, and expiry.

Design constraints:
  - Standard library, unittest, and fastapi.testclient only.
  - Environment modified exclusively via patch.dict(os.environ, {...}, clear=True).
  - Never import main.py.
  - Pure ASCII source: any non-ASCII characters are written as \\uXXXX escapes.
  - No __main__ entrypoint.
"""

import importlib
import os
import re
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import services.security as security
from services.security import (
    API_KEY_HEADER,
    PUBLIC_PATHS,
    SecurityConfigError,
    _digest,
    _get_secret,
    make_oauth_state,
    require_api_key,
    verify_oauth_state,
)

# Sample valid shared secret (>= 24 characters).
_VALID_SECRET = "super-secret-key-at-least-24-characters-long"
_VALID_SECRET_ALT = "another-valid-secret-key-with-24-plus-chars"


# ---------------------------------------------------------------------------
# TEST APP HELPER
# ---------------------------------------------------------------------------

def _create_throwaway_app() -> FastAPI:
    """Create a minimal throwaway FastAPI application for testing security rules.

    Purpose:
        Builds an isolated FastAPI instance configured with the require_api_key
        dependency to verify authentication behavior across public and protected
        routes without touching application state or importing main.py.

    Parameters:
        None.

    Returns:
        FastAPI: A configured application with standard test routes.

    Raises:
        None.
    """
    app = FastAPI(dependencies=[Depends(require_api_key)])

    @app.get("/")
    def get_root():
        """Public root route."""
        return {"status": "ok", "route": "/"}

    @app.get("/health")
    def get_health():
        """Public health check route."""
        return {"status": "healthy"}

    @app.get("/leads/test")
    def get_leads_test():
        """Protected leads inspection route."""
        return {"status": "ok", "route": "/leads/test"}

    @app.post("/email/send")
    def post_email_send():
        """Protected outbound email route."""
        return {"status": "ok", "route": "/email/send"}

    @app.get("/email/auth/callback")
    def get_email_auth_callback():
        """Public OAuth callback route."""
        return {"status": "ok", "route": "/email/auth/callback"}

    return app


# ---------------------------------------------------------------------------
# A. Import safety
# ---------------------------------------------------------------------------

class TestAImportSafety(unittest.TestCase):
    """Verify import-time safety and source integrity of services/security.py."""

    def test_a01_import_with_empty_environment(self) -> None:
        """Ensure services/security.py imports cleanly with an empty environment.

        Purpose:
            Verifies that module loading does not read os.environ at import time,
            ensuring no exceptions are raised in unconfigured environments.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If reloading services.security raises an exception.
        """
        with patch.dict(os.environ, {}, clear=True):
            try:
                importlib.reload(security)
            except Exception as exc:
                self.fail(f"Reloading services.security with empty env raised {exc!r}")

    def test_a02_source_decodes_as_ascii(self) -> None:
        """Ensure services/security.py contains only ASCII characters.

        Purpose:
            Guarantees source code portability and prevents encoding issues
            by verifying that the file decodes strictly as 7-bit ASCII.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If non-ASCII bytes are found in the source file.
        """
        source_path = Path(__file__).resolve().parent.parent / "services" / "security.py"
        raw_bytes = source_path.read_bytes()
        try:
            raw_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"services/security.py contains non-ASCII characters: {exc}")

    def test_a03_source_contains_compare_digest(self) -> None:
        """Ensure services/security.py uses hmac.compare_digest for constant-time checks.

        Purpose:
            Validates that the source code employs constant-time comparison
            functions to prevent timing attacks on keys and signatures.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If 'compare_digest' is absent from the source.
        """
        source_path = Path(__file__).resolve().parent.parent / "services" / "security.py"
        source_text = source_path.read_text(encoding="utf-8")
        self.assertIn(
            "compare_digest",
            source_text,
            "services/security.py must use compare_digest for timing-safe checks",
        )


# ---------------------------------------------------------------------------
# B. Throwaway app authentication & API key enforcement
# ---------------------------------------------------------------------------

class TestBThrowawayApp(unittest.TestCase):
    """Verify HTTP API key authentication behaviors on a throwaway FastAPI app."""

    def setUp(self) -> None:
        """Initialize the throwaway test client before each test.

        Purpose:
            Prepares an isolated TestClient instance linked to the throwaway app.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.app = _create_throwaway_app()
        self.client = TestClient(self.app)

    def test_b01_secret_unset_protected_503_and_public_200(self) -> None:
        """Verify that an unset secret returns 503 for protected and 200 for public routes.

        Purpose:
            Ensures fail-closed behavior: without a configured secret, protected
            routes return 503 while public endpoints remain accessible.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If response status codes or details do not match.
        """
        with patch.dict(os.environ, {}, clear=True):
            # Public routes must succeed
            for path in ["/", "/health", "/email/auth/callback"]:
                resp = self.client.get(path)
                self.assertEqual(
                    resp.status_code,
                    200,
                    f"Public route {path!r} must return 200 when secret is unset",
                )

            # Protected routes must return 503 Service Unavailable
            for path, method in [("/leads/test", "get"), ("/email/send", "post")]:
                caller = getattr(self.client, method)
                resp = caller(path)
                self.assertEqual(
                    resp.status_code,
                    503,
                    f"Protected route {path!r} must return 503 when secret is unset",
                )
                self.assertEqual(
                    resp.json().get("detail"),
                    "API authentication is not configured.",
                )

    def test_b02_secret_23_characters_gives_503(self) -> None:
        """Verify that an invalid secret shorter than 24 characters returns 503.

        Purpose:
            Confirms minimum key length enforcement (>= 24 characters). Secrets
            with 23 characters must fail closed with HTTP 503.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If short secrets do not trigger 503 on protected routes.
        """
        short_secret = "12345678901234567890123"  # exactly 23 characters
        self.assertEqual(len(short_secret), 23)

        with patch.dict(os.environ, {"API_SECRET_KEY": short_secret}, clear=True):
            resp = self.client.get("/leads/test")
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(
                resp.json().get("detail"),
                "API authentication is not configured.",
            )

            resp_with_key = self.client.get(
                "/leads/test", headers={"X-API-Key": short_secret}
            )
            self.assertEqual(resp_with_key.status_code, 503)

    def test_b03_no_header_gives_401_with_www_authenticate(self) -> None:
        """Verify that missing API key header returns 401 with WWW-Authenticate.

        Purpose:
            Ensures unauthenticated requests to protected routes receive HTTP 401
            along with standard WWW-Authenticate: ApiKey challenge headers.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If status code, challenge header, or detail mismatch.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            resp = self.client.get("/leads/test")
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(
                resp.headers.get("www-authenticate"),
                "ApiKey",
                "Missing API key must return WWW-Authenticate: ApiKey header",
            )
            self.assertEqual(
                resp.json().get("detail"),
                "Invalid or missing API key.",
            )

    def test_b04_empty_header_gives_401(self) -> None:
        """Verify that empty or whitespace-only API key headers return 401.

        Purpose:
            Confirms that blank headers are treated as unauthenticated and
            consistently rejected with HTTP 401 and WWW-Authenticate.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If empty or blank headers do not return 401.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            for blank_val in ["", "   ", "\t"]:
                resp = self.client.get(
                    "/leads/test", headers={API_KEY_HEADER: blank_val}
                )
                self.assertEqual(
                    resp.status_code,
                    401,
                    f"Blank header {blank_val!r} must return 401",
                )
                self.assertEqual(resp.headers.get("www-authenticate"), "ApiKey")
                self.assertEqual(
                    resp.json().get("detail"),
                    "Invalid or missing API key.",
                )

    def test_b05_wrong_key_same_length_gives_401(self) -> None:
        """Verify that wrong keys of identical length return 401.

        Purpose:
            Checks constant-time comparison behavior: keys with identical length
            differing by a single character must be rejected with HTTP 401.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If wrong key fails to produce 401.
        """
        wrong_key = _VALID_SECRET[:-1] + ("x" if _VALID_SECRET[-1] != "x" else "y")
        self.assertEqual(len(wrong_key), len(_VALID_SECRET))

        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            resp = self.client.get("/leads/test", headers={API_KEY_HEADER: wrong_key})
            self.assertEqual(resp.status_code, 401)
            self.assertEqual(resp.headers.get("www-authenticate"), "ApiKey")
            self.assertEqual(
                resp.json().get("detail"),
                "Invalid or missing API key.",
            )

    def test_b06_correct_key_gives_200_including_lowercase_header(self) -> None:
        """Verify that correct API keys return 200, including case-insensitive headers.

        Purpose:
            Verifies that authentic requests succeed on protected routes and
            that header lookup supports standard case-insensitive naming.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If valid key does not produce HTTP 200.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            # Standard case header
            resp_get = self.client.get(
                "/leads/test", headers={"X-API-Key": _VALID_SECRET}
            )
            self.assertEqual(resp_get.status_code, 200)

            # Lowercase header
            resp_lower = self.client.get(
                "/leads/test", headers={"x-api-key": _VALID_SECRET}
            )
            self.assertEqual(resp_lower.status_code, 200)

            # POST route with lowercase header
            resp_post = self.client.post(
                "/email/send", headers={"x-api-key": _VALID_SECRET}
            )
            self.assertEqual(resp_post.status_code, 200)

    def test_b07_non_ascii_header_value_gives_401_never_500(self) -> None:
        """Verify that non-ASCII header values return 401 and never cause HTTP 500.

        Purpose:
            Ensures robustness against malformed or multi-byte characters in
            incoming HTTP headers. Digesting must use replacement to prevent
            encoding crashes.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If non-ASCII headers trigger 500 or non-401 status.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            # 1. Via raw bytes header in TestClient
            resp = self.client.get(
                "/leads/test",
                headers=[(b"x-api-key", b"non-ascii-\xff\xfe-key-probe")],
            )
            self.assertEqual(resp.status_code, 401)
            self.assertNotEqual(resp.status_code, 500)

            # 2. Via direct dependency invocation with unicode string
            dummy_request = Request(
                {"type": "http", "method": "GET", "path": "/leads/test", "headers": []}
            )
            with self.assertRaises(HTTPException) as ctx:
                require_api_key(
                    dummy_request,
                    provided="non-ascii-probe-\\u00e9\\u00e0\\u00fc",
                )
            self.assertEqual(ctx.exception.status_code, 401)

    def test_b08_secret_and_provided_key_never_leak(self) -> None:
        """Ensure secret keys and provided keys never appear in response bodies or headers.

        Purpose:
            Confirms zero-leakage security rule: neither the shared secret nor
            any candidate key supplied by a client may ever be reflected in HTTP
            response bodies or headers.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If secret or provided key is found in any response part.
        """
        secret = "secret-token-cannot-leak-into-any-body-or-header"
        provided = "provided-key-cannot-leak-into-any-body-or-header"

        with patch.dict(os.environ, {"API_SECRET_KEY": secret}, clear=True):
            responses = [
                self.client.get("/leads/test", headers={API_KEY_HEADER: provided}),
                self.client.get("/leads/test", headers={API_KEY_HEADER: secret}),
                self.client.get("/leads/test"),
            ]

        with patch.dict(os.environ, {}, clear=True):
            responses.append(
                self.client.get("/leads/test", headers={API_KEY_HEADER: provided})
            )

        for resp in responses:
            self.assertNotIn(secret, resp.text, "Secret key was leaked in response body")
            self.assertNotIn(provided, resp.text, "Provided key was leaked in response body")
            for header_name, header_value in resp.headers.items():
                self.assertNotIn(
                    secret,
                    header_name,
                    "Secret key was leaked in response header name",
                )
                self.assertNotIn(
                    secret,
                    header_value,
                    "Secret key was leaked in response header value",
                )
                self.assertNotIn(
                    provided,
                    header_name,
                    "Provided key was leaked in response header name",
                )
                self.assertNotIn(
                    provided,
                    header_value,
                    "Provided key was leaked in response header value",
                )

    def test_b09_secret_read_at_call_time_changing_env_changes_result(self) -> None:
        """Verify that the shared secret is read dynamically at call time.

        Purpose:
            Ensures that rotating API_SECRET_KEY in os.environ takes effect
            immediately on subsequent calls without process restart or reload.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If rotating environment secret does not change results.
        """
        secret_a = _VALID_SECRET
        secret_b = _VALID_SECRET_ALT

        with patch.dict(os.environ, {"API_SECRET_KEY": secret_a}, clear=True):
            # secret_a is accepted; secret_b is rejected
            r1 = self.client.get("/leads/test", headers={API_KEY_HEADER: secret_a})
            self.assertEqual(r1.status_code, 200)

            r2 = self.client.get("/leads/test", headers={API_KEY_HEADER: secret_b})
            self.assertEqual(r2.status_code, 401)

            # Rotate secret in place
            os.environ["API_SECRET_KEY"] = secret_b

            # secret_a is now rejected; secret_b is accepted
            r3 = self.client.get("/leads/test", headers={API_KEY_HEADER: secret_a})
            self.assertEqual(r3.status_code, 401)

            r4 = self.client.get("/leads/test", headers={API_KEY_HEADER: secret_b})
            self.assertEqual(r4.status_code, 200)


# ---------------------------------------------------------------------------
# C. Public paths whitelist
# ---------------------------------------------------------------------------

class TestCPublicPaths(unittest.TestCase):
    """Verify exact membership and exclusions of the PUBLIC_PATHS set."""

    def test_c01_public_paths_exact_set(self) -> None:
        """Verify that PUBLIC_PATHS contains exactly the approved public paths.

        Purpose:
            Ensures that only /, /health, and /email/auth/callback are whitelisted.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If PUBLIC_PATHS does not match the exact set.
        """
        expected = frozenset({"/", "/health", "/email/auth/callback"})
        self.assertEqual(
            PUBLIC_PATHS,
            expected,
            f"PUBLIC_PATHS must be exactly {expected!r}, got {PUBLIC_PATHS!r}",
        )

    def test_c02_non_public_paths_are_not_whitelisted(self) -> None:
        """Verify that related but protected paths are not inadvertently public.

        Purpose:
            Prevents path prefix confusion attacks by ensuring routes such as
            /email/auth or /email/send are not classified as public.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If any protected route is present in PUBLIC_PATHS.
        """
        forbidden_in_public = [
            "/email/auth",
            "/email/send",
            "/leads/test",
            "/leads",
            "/healthcheck",
            "",
        ]
        for path in forbidden_in_public:
            self.assertNotIn(
                path,
                PUBLIC_PATHS,
                f"Path {path!r} must never be in PUBLIC_PATHS",
            )


# ---------------------------------------------------------------------------
# D. OAuth state token management
# ---------------------------------------------------------------------------

class TestDState(unittest.TestCase):
    """Verify generation, structural constraints, and validation of OAuth state tokens."""

    def test_d01_round_trip_is_true(self) -> None:
        """Verify that a freshly generated OAuth state token validates successfully.

        Purpose:
            Confirms end-to-end functionality of make_oauth_state and
            verify_oauth_state under normal operating conditions.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If a newly created state fails verification.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            state = make_oauth_state()
            self.assertTrue(
                verify_oauth_state(state),
                "Freshly generated OAuth state token must verify as True",
            )

    def test_d02_two_states_differ(self) -> None:
        """Verify that consecutive calls to make_oauth_state produce distinct tokens.

        Purpose:
            Confirms cryptographically secure randomness in state generation,
            preventing token collisions and replay attacks.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If two generated states are identical.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            s1 = make_oauth_state()
            s2 = make_oauth_state()
            self.assertNotEqual(s1, s2, "Consecutive state tokens must be unique")

    def test_d03_matches_regex_character_set(self) -> None:
        """Verify that OAuth state tokens contain strictly URL-safe characters.

        Purpose:
            Ensures that tokens conform to ^[A-Za-z0-9_.-]+$ for safe inclusion
            in query parameters without URL encoding issues.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If a token contains non-URL-safe characters.
        """
        pattern = re.compile(r"^[A-Za-z0-9_.-]+$")
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            state = make_oauth_state()
            self.assertRegex(
                state,
                pattern,
                "OAuth state token must contain only URL-safe [A-Za-z0-9_.-] characters",
            )

    def test_d04_one_flipped_character_returns_false(self) -> None:
        """Verify that tampering with any character in a state token causes validation to fail.

        Purpose:
            Confirms HMAC integrity checking: modifying the timestamp, random
            payload, or signature must result in verify_oauth_state returning False.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If a modified token passes verification.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            state = make_oauth_state()

            # Flip the last character (signature segment)
            flipped_last = state[:-1] + ("A" if state[-1] != "A" else "B")
            self.assertFalse(
                verify_oauth_state(flipped_last),
                "Token with flipped signature character must fail verification",
            )

            # Flip a character in the token payload segment
            parts = state.split(".")
            self.assertEqual(len(parts), 3)
            token_part = parts[1]
            flipped_token = token_part[:-1] + ("X" if token_part[-1] != "X" else "Y")
            tampered_state = f"{parts[0]}.{flipped_token}.{parts[2]}"
            self.assertFalse(
                verify_oauth_state(tampered_state),
                "Token with flipped payload character must fail verification",
            )

    def test_d05_verified_under_different_secret_returns_false(self) -> None:
        """Verify that tokens signed with one secret cannot be verified under another.

        Purpose:
            Ensures secret isolation: rotating or changing secrets invalidates
            previously issued state tokens.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If a token validates under an incorrect secret key.
        """
        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            state = make_oauth_state()

        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET_ALT}, clear=True):
            self.assertFalse(
                verify_oauth_state(state),
                "Token signed under secret A must fail verification under secret B",
            )

    def test_d06_expired_state_returns_false(self) -> None:
        """Verify that state tokens older than _STATE_TTL_SECONDS (600s) expire.

        Purpose:
            Enforces token time-to-live expiration. Tokens aged 601s must fail,
            while tokens at or within 600s must remain valid.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If expired tokens pass or fresh tokens fail.
        """
        t0 = 1700000000.0

        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            with patch("time.time", return_value=t0):
                state = make_oauth_state()

            # 601 seconds later -> expired
            with patch("time.time", return_value=t0 + 601.0):
                self.assertFalse(
                    verify_oauth_state(state),
                    "State token aged 601s must be expired and return False",
                )

            # Exactly 600 seconds later -> boundary condition, valid
            with patch("time.time", return_value=t0 + 600.0):
                self.assertTrue(
                    verify_oauth_state(state),
                    "State token aged exactly 600s must remain valid",
                )

    def test_d07_future_state_beyond_skew_returns_false(self) -> None:
        """Verify that timestamps more than _STATE_SKEW_SECONDS (60s) in the future fail.

        Purpose:
            Prevents pre-generated future token attacks while allowing reasonable
            clock skew (up to 60s).

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If tokens beyond allowed clock skew pass verification.
        """
        t0 = 1700000000.0

        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            with patch("time.time", return_value=t0):
                state = make_oauth_state()

            # Clock moved backward by 61s (state timestamp is now 61s in future)
            with patch("time.time", return_value=t0 - 61.0):
                self.assertFalse(
                    verify_oauth_state(state),
                    "Token with timestamp 61s in the future must fail verification",
                )

            # Clock moved backward by 60s (acceptable skew boundary)
            with patch("time.time", return_value=t0 - 60.0):
                self.assertTrue(
                    verify_oauth_state(state),
                    "Token with timestamp 60s in the future must be accepted within skew",
                )

    def test_d08_malformed_inputs_return_false_without_raising(self) -> None:
        """Verify that invalid and malformed inputs return False and never raise exceptions.

        Purpose:
            Confirms total fault-tolerance in verify_oauth_state across non-string,
            empty, truncated, oversized, and structurally invalid inputs.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If any malformed input returns True or raises an unhandled exception.
        """
        malformed_cases = [
            None,
            "",
            "a",
            "a.b",
            "a.b.c.d",
            123,
            "x" * 10000,
            [],
            {},
            3.14,
            "not_a_number.token.signature",
            "1700000000..signature",
            "1700000000.token.",
        ]

        with patch.dict(os.environ, {"API_SECRET_KEY": _VALID_SECRET}, clear=True):
            for candidate in malformed_cases:
                try:
                    result = verify_oauth_state(candidate)
                    self.assertFalse(
                        result,
                        f"Malformed candidate {candidate!r} must return False",
                    )
                except Exception as exc:
                    self.fail(
                        f"verify_oauth_state({candidate!r}) raised unhandled exception: {exc}"
                    )

    def test_d09_secret_unset_behavior(self) -> None:
        """Verify state creation and verification behavior when secret is unset or short.

        Purpose:
            Confirms that make_oauth_state raises SecurityConfigError when the
            environment is improperly configured, while verify_oauth_state safely
            returns False without raising.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If SecurityConfigError is not raised by make_oauth_state,
                or if verify_oauth_state raises an exception.
        """
        # Secret completely unset
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(security.SecurityConfigError):
                make_oauth_state()

            self.assertFalse(
                verify_oauth_state("1700000000.tokendata.signature"),
                "verify_oauth_state must return False when API_SECRET_KEY is unset",
            )

        # Secret is 23 characters (too short)
        short_key = "12345678901234567890123"
        with patch.dict(os.environ, {"API_SECRET_KEY": short_key}, clear=True):
            with self.assertRaises(security.SecurityConfigError):
                make_oauth_state()

            self.assertFalse(
                verify_oauth_state("1700000000.tokendata.signature"),
                "verify_oauth_state must return False when API_SECRET_KEY is too short",
            )
