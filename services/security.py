"""
services/security.py

Security and authentication layer for the real-estate lead platform.

Responsibilities:
  - Enforce API key authentication across protected FastAPI routes.
  - Fail closed when the shared secret is unconfigured or invalid.
  - Generate and verify tamper-resistant, timestamped OAuth state tokens.
  - Constant-time secret and token comparisons to mitigate timing attacks.

Design constraints:
  - Standard library and FastAPI only; no third-party cryptography packages.
  - The shared secret is read ONLY inside functions, at call time, from
    API_SECRET_KEY (stripped). Nothing is read at import time.
  - Safe to import with an empty environment.
  - Pure ASCII source: any non-ASCII characters are written as \\uXXXX escapes.
"""

import base64
import hashlib
import hmac
import os
import secrets
import time

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader


# ---------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS (no env reads here)
# ---------------------------------------------------------------------------

API_KEY_HEADER: str = "X-API-Key"
_MIN_KEY_LENGTH: int = 24
_STATE_TTL_SECONDS: int = 600
_STATE_SKEW_SECONDS: int = 60
PUBLIC_PATHS: frozenset[str] = frozenset({"/", "/health", "/email/auth/callback"})

api_key_header: APIKeyHeader = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    scheme_name="ApiKeyAuth",
)


# ---------------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------------

class SecurityConfigError(RuntimeError):
    """Exception raised when the API security configuration is invalid or missing.

    Purpose:
        Signals that API_SECRET_KEY is unset, empty, or does not meet the
        minimum length requirement of 24 characters.

    Parameters:
        message (str): Explanatory error message.

    Returns:
        None.

    Raises:
        None.
    """


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _get_secret() -> str:
    """Retrieve and validate the shared API secret key from the environment.

    Purpose:
        Reads API_SECRET_KEY from environment variables at call time, strips
        surrounding whitespace, and verifies that it is at least 24 characters.

    Parameters:
        None.

    Returns:
        str: The stripped secret key.

    Raises:
        SecurityConfigError: If API_SECRET_KEY is missing, blank, or shorter
            than _MIN_KEY_LENGTH (24 characters).
    """
    secret = os.environ.get("API_SECRET_KEY", "").strip()
    if len(secret) < _MIN_KEY_LENGTH:
        raise SecurityConfigError(
            "API_SECRET_KEY is missing or shorter than 24 characters."
        )
    return secret


def _digest(value: str) -> bytes:
    """Compute the SHA-256 digest of a string value.

    Purpose:
        Produces a fixed 32-byte digest of the given string to enable
        constant-time comparisons via hmac.compare_digest on equal-length
        buffers, mitigating timing attacks.

    Parameters:
        value (str): The string to hash. Encoded using UTF-8 with replacement
            to handle non-ASCII or malformed input without raising errors.

    Returns:
        bytes: The 32-byte SHA-256 digest.

    Raises:
        None.
    """
    return hashlib.sha256(value.encode("utf-8", "replace")).digest()


# ---------------------------------------------------------------------------
# FASTAPI AUTHENTICATION DEPENDENCY
# ---------------------------------------------------------------------------

def require_api_key(
    request: Request,
    provided: str | None = Security(api_key_header),
) -> None:
    """FastAPI route dependency enforcing API key authentication.

    Purpose:
        Authenticates incoming HTTP requests against the shared secret key.
        Allows exact matches in PUBLIC_PATHS without authentication.
        Fails closed with HTTP 503 if the secret is unconfigured or invalid.
        Rejects missing, blank, or invalid keys with HTTP 401.

    Parameters:
        request (Request): The incoming FastAPI/Starlette request object.
        provided (str | None): The API key extracted from the request header
            (defaults to Security(api_key_header)).

    Returns:
        None.

    Raises:
        HTTPException:
            - 503 (Service Unavailable) if API_SECRET_KEY is not configured
              or is shorter than 24 characters.
            - 401 (Unauthorized) with WWW-Authenticate: ApiKey header if the
              API key is missing, blank, or does not match the shared secret.
    """
    # 1. request.url.path in PUBLIC_PATHS -> return (exact match only).
    if request.url.path in PUBLIC_PATHS:
        return

    # 2. _get_secret(); SecurityConfigError -> HTTPException 503
    try:
        secret = _get_secret()
    except SecurityConfigError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    # 3. provided None or blank -> HTTPException 401, header WWW-Authenticate: ApiKey
    if provided is None or not provided.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # 4. hmac.compare_digest(_digest(provided), _digest(secret)) false -> the same 401.
    if not hmac.compare_digest(_digest(provided), _digest(secret)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------------------------------------------------------------------------
# OAUTH STATE MANAGEMENT
# ---------------------------------------------------------------------------

def make_oauth_state() -> str:
    """Generate a signed, timestamped, URL-safe OAuth state token.

    Purpose:
        Creates a tamper-resistant state token for OAuth flows. The token
        consists of a timestamp and 16 random bytes signed with an HMAC-SHA256
        key derived from the shared secret key.

    Parameters:
        None.

    Returns:
        str: A URL-safe state string in the format
            "<timestamp>.<token>.<signature>" using only [A-Za-z0-9_.-].

    Raises:
        SecurityConfigError: If API_SECRET_KEY is unset, blank, or shorter
            than 24 characters.
    """
    secret = _get_secret()
    payload = f"{int(time.time())}." + secrets.token_urlsafe(16)
    signing_key = hmac.new(
        secret.encode("utf-8"), b"oauth-state-v1", hashlib.sha256
    ).digest()
    sig_raw = hmac.new(
        signing_key, payload.encode("utf-8"), hashlib.sha256
    ).digest()
    signature = base64.urlsafe_b64encode(sig_raw).decode("ascii").rstrip("=")
    return f"{payload}.{signature}"


def verify_oauth_state(state: object) -> bool:
    """Verify an OAuth state token for authenticity, structure, and freshness.

    Purpose:
        Validates that a given state token was produced by make_oauth_state
        using the current secret key, is structurally well-formed, has not
        expired (age <= _STATE_TTL_SECONDS), and is not in the future beyond
        allowable clock skew (future delta <= _STATE_SKEW_SECONDS). Never
        raises an exception.

    Parameters:
        state (object): The candidate state token to verify.

    Returns:
        bool: True if the state token is authentic, structurally valid,
            and within acceptable time boundaries; False otherwise.

    Raises:
        None: Never raises any exception under any circumstances.
    """
    try:
        if not isinstance(state, str):
            return False

        parts = state.split(".")
        if len(parts) != 3:
            return False

        timestamp_str, token, signature = parts
        if not timestamp_str or not token or not signature:
            return False

        ts = int(timestamp_str)
        now = time.time()
        age = now - ts
        if age > _STATE_TTL_SECONDS:
            return False
        if ts - now > _STATE_SKEW_SECONDS:
            return False

        secret = _get_secret()
        payload = f"{timestamp_str}.{token}"
        signing_key = hmac.new(
            secret.encode("utf-8"), b"oauth-state-v1", hashlib.sha256
        ).digest()
        expected_sig_raw = hmac.new(
            signing_key, payload.encode("utf-8"), hashlib.sha256
        ).digest()
        expected_sig = (
            base64.urlsafe_b64encode(expected_sig_raw)
            .decode("ascii")
            .rstrip("=")
        )

        return hmac.compare_digest(_digest(signature), _digest(expected_sig))
    except Exception:
        return False
