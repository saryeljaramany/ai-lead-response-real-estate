"""
services/captcha_service.py

Cloudflare Turnstile CAPTCHA verification service for the real-estate lead system.

This module provides server-side verification of Cloudflare Turnstile tokens
acting as a fail-closed gate in front of lead intake. It is designed so that:

    - Importing the module with an empty environment never raises.
    - Environment variables are read only inside functions at call time.
    - No secret, token, IP address, or provider response text is ever leaked
      in exceptions, logs, or return values.
    - Every failure is classified into a category (config or transient) or
      surfaced with a specific fail-closed reason.
"""

from __future__ import annotations

import ipaddress
import os
import re
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERIFY_URL: str = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
_REQUEST_TIMEOUT_SECONDS: int = 10
_MAX_TOKEN_LENGTH: int = 2048
_MAX_ERROR_CODES: int = 10
_ALLOWED_CODE_CHARS = re.compile(r"[^A-Za-z0-9_-]")

# Environment variable names read dynamically at call time
_ENV_SECRET_KEY: str = "TURNSTILE_SECRET_KEY"
_ENV_EXPECTED_HOSTNAME: str = "TURNSTILE_EXPECTED_HOSTNAME"


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class CaptchaError(RuntimeError):
    """Classified CAPTCHA integration error.

    Parameters
    ----------
    message : str
        Fixed diagnostic text without secrets, tokens, or PII.
    category : str
        One of ``"config"`` or ``"transient"``.
    """

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category: str = category


# ---------------------------------------------------------------------------
# Private helpers -- environment and input sanitisation
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Read an environment variable at call time and return its stripped value.

    Parameters
    ----------
    name : str
        The name of the environment variable to read.

    Returns
    -------
    str
        The stripped, non-empty value.

    Raises
    ------
    CaptchaError
        With ``category="config"`` if the variable is unset or blank.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        raise CaptchaError(
            f"{name} is missing or empty. Set it in .env and restart the server.",
            category="config",
        )
    return value.strip()


def _expected_hostname() -> str | None:
    """Read optional expected hostname from environment.

    Returns
    -------
    str or None
        Stripped and lower-cased hostname, or None if unset or blank.
    """
    val = os.getenv(_ENV_EXPECTED_HOSTNAME)
    if val is None or not val.strip():
        return None
    return val.strip().lower()


def _get_http_client() -> httpx.Client:
    """Create a fresh HTTP client configured for Turnstile verification.

    Returns
    -------
    httpx.Client
        A fresh client with timeout set, intended to be used as a context
        manager and never cached.
    """
    return httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS)


def _clean_codes(value: object) -> list[str]:
    """Clean and filter error codes from the provider response.

    Parameters
    ----------
    value : object
        The raw error-codes value from the provider response.

    Returns
    -------
    list of str
        List of cleaned, alphanumeric/hyphen/underscore error codes up to
        64 characters each, capped at 10 items.
    """
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        filtered = _ALLOWED_CODE_CHARS.sub("", item)[:64]
        if filtered:
            cleaned.append(filtered)
            if len(cleaned) == _MAX_ERROR_CODES:
                break
    return cleaned


def _clean_remote_ip(value: object) -> str | None:
    """Validate and clean a remote IP address.

    Parameters
    ----------
    value : object
        The candidate IP address object to validate.

    Returns
    -------
    str or None
        Stripped valid IPv4 or IPv6 address string, or None if invalid.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        ipaddress.ip_address(stripped)
        return stripped
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_token(token: object, remote_ip: object = None) -> dict[str, Any]:
    """Verify a Cloudflare Turnstile token server-side.

    Parameters
    ----------
    token : object
        The CAPTCHA response token to verify.
    remote_ip : object, optional
        The client's remote IP address, if available.

    Returns
    -------
    dict
        Verification result containing ``verified`` (bool), ``reason`` (str),
        and ``error_codes`` (list of str).

    Raises
    ------
    CaptchaError
        With ``category="config"`` or ``category="transient"`` on provider
        errors or misconfiguration.
    """
    secret = _require_env(_ENV_SECRET_KEY)

    if not isinstance(token, str):
        return {"verified": False, "reason": "missing_token", "error_codes": []}
    stripped_token = token.strip()
    if not stripped_token:
        return {"verified": False, "reason": "missing_token", "error_codes": []}

    if len(stripped_token) > _MAX_TOKEN_LENGTH:
        return {"verified": False, "reason": "invalid_token", "error_codes": []}

    form_data: dict[str, str] = {
        "secret": secret,
        "response": stripped_token,
    }
    cleaned_ip = _clean_remote_ip(remote_ip)
    if cleaned_ip is not None:
        form_data["remoteip"] = cleaned_ip

    try:
        with _get_http_client() as client:
            response = client.post(_VERIFY_URL, data=form_data)
    except httpx.HTTPError:
        raise CaptchaError(
            "Could not reach the CAPTCHA provider (network error or timeout).",
            category="transient",
        ) from None

    status = response.status_code
    if status == 429 or status >= 500:
        raise CaptchaError(
            "The CAPTCHA provider is temporarily unavailable. Retry later.",
            category="transient",
        )
    if status != 200:
        raise CaptchaError(
            f"The CAPTCHA provider rejected the request (HTTP {status}).",
            category="config",
        )

    try:
        body = response.json()
    except Exception:
        raise CaptchaError(
            "The CAPTCHA provider returned an unreadable response.",
            category="transient",
        ) from None

    if not isinstance(body, dict):
        raise CaptchaError(
            "The CAPTCHA provider returned an unreadable response.",
            category="transient",
        )

    if body.get("success") is True:
        expected = _expected_hostname()
        if expected is not None:
            raw_host = body.get("hostname")
            if not isinstance(raw_host, str) or raw_host.strip().lower() != expected:
                return {"verified": False, "reason": "hostname_mismatch", "error_codes": []}
        return {"verified": True, "reason": "ok", "error_codes": []}

    codes = _clean_codes(body.get("error-codes"))
    if "missing-input-secret" in codes or "invalid-input-secret" in codes:
        raise CaptchaError(
            "The CAPTCHA secret key was rejected by the provider. Check TURNSTILE_SECRET_KEY.",
            category="config",
        )
    if "internal-error" in codes:
        raise CaptchaError(
            "The CAPTCHA provider reported an internal error. Retry later.",
            category="transient",
        )
    if "timeout-or-duplicate" in codes:
        return {"verified": False, "reason": "expired_or_used", "error_codes": codes}
    if "missing-input-response" in codes or "invalid-input-response" in codes:
        return {"verified": False, "reason": "invalid_token", "error_codes": codes}
    return {"verified": False, "reason": "rejected", "error_codes": codes}
