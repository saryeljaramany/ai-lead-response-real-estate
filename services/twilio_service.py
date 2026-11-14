"""
services/twilio_service.py

Twilio SMS integration layer for the AI-powered lead management system.

This module receives an already-validated SMSPayload and returns a result
dict. It is designed around three guarantees:

    1. Import safety. Importing this module with an empty environment never
       raises. No environment variable is read and no Twilio client is built
       at import time; everything happens at call time, after load_dotenv()
       in main.py has run.
    2. Dry-run by default. Nothing is sent to Twilio unless TWILIO_DRY_RUN is
       explicitly set to false, 0, no or off. A missing or mistyped value
       keeps the service in dry-run.
    3. Classified failures. Every Twilio or network failure is converted into
       an SMSSendError whose category tells the HTTP layer and the n8n
       pipeline whether a retry makes sense.

Validation (E.164 phone normalisation and GSM-7 body sanitising) runs
identically in dry-run and live mode, so dry-run is a faithful rehearsal.

Environment variables (all read inside functions):
    TWILIO_ACCOUNT_SID             live mode only
    TWILIO_AUTH_TOKEN              live mode only
    TWILIO_PHONE_NUMBER            live mode, unless a Messaging Service is set
    TWILIO_MESSAGING_SERVICE_SID   optional; takes precedence over the number
    TWILIO_DRY_RUN                 see above
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import math
import os
import re

import requests
from twilio.base.exceptions import TwilioRestException
from twilio.http.http_client import TwilioHttpClient
from twilio.rest import Client

from models import SMSPayload


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The Twilio SDK sets no request timeout by default, so one is mandatory.
_REQUEST_TIMEOUT_SECONDS = 10

# E.164: a plus sign, a non-zero first digit, 8 to 15 digits in total.
_E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")

# GSM 03.38 default alphabet, without the escape code. Each character
# counts as one septet. Non-ASCII characters are written as unicode escapes
# so that this file stays pure ASCII.
_GSM7_BASIC = frozenset(
    "@$_ !\"#%&'()*+,-./0123456789:;<=>?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "\n\r"
    "\u00a3\u00a5\u00e8\u00e9\u00f9\u00ec\u00f2\u00c7\u00d8\u00f8\u00c5\u00e5"
    "\u0394\u03a6\u0393\u039b\u03a9\u03a0\u03a8\u03a3\u0398\u039e"
    "\u00c6\u00e6\u00df\u00c9\u00a4\u00a1\u00c4\u00d6\u00d1\u00dc\u00a7\u00bf"
    "\u00e4\u00f6\u00f1\u00fc\u00e0"
)

# GSM 03.38 extension table. Each character is sent as an escape plus the
# character, so it counts as two septets.
_GSM7_EXTENSION = frozenset("^{}\\[~]|\u20ac\f")

# Typographic characters that LLM output is full of, mapped to GSM-7-safe
# equivalents. Keys are code points, as required by str.translate. An empty
# string value deletes the character.
_PUNCTUATION_MAP = {
    # Single curly quotes and low/reversed variants -> apostrophe
    **dict.fromkeys((0x2018, 0x2019, 0x201A, 0x201B), "'"),
    # Double curly quotes and low/reversed variants -> straight double quote
    **dict.fromkeys((0x201C, 0x201D, 0x201E, 0x201F), '"'),
    # En dash, em dash, horizontal bar, minus sign -> hyphen
    **dict.fromkeys((0x2013, 0x2014, 0x2015, 0x2212), "-"),
    # Ellipsis -> three dots
    0x2026: "...",
    # Bullet -> hyphen
    0x2022: "-",
    # No-break space, narrow no-break space, thin space -> plain space
    **dict.fromkeys((0x00A0, 0x202F, 0x2009), " "),
    # Zero-width space and byte order mark -> deleted
    **dict.fromkeys((0x200B, 0xFEFF), ""),
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class SMSSendError(RuntimeError):
    """
    Raised when an SMS cannot be sent, carrying a retry-relevant category.

    Attributes:
        category: One of three values.
            "recipient" - the destination cannot be messaged; never retry.
            "config"    - a sender, account or server setup problem that
                          fails for every recipient; never retry, fix the
                          configuration instead.
            "transient" - Twilio or the network failed; a retry is
                          reasonable.
        twilio_code: The Twilio error code when the failure came from a
            Twilio API response, otherwise None.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        twilio_code: int | None = None,
    ) -> None:
        """
        Create the error.

        Parameters:
            message: Actionable, human-readable description of the failure.
            category: "recipient", "config" or "transient".
            twilio_code: Optional Twilio error code.
        """
        super().__init__(message)
        self.category = category
        self.twilio_code = twilio_code


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """
    Read an environment variable at call time, strip it, and return it.

    Parameters:
        name: The environment variable name to look up.

    Returns:
        The stripped, non-empty value of the variable.

    Raises:
        SMSSendError: category "config", if the variable is unset, empty or
            whitespace only.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise SMSSendError(
            f"{name} is missing or empty. Set it in .env and restart the server.",
            category="config",
        )
    return value


def _is_dry_run() -> bool:
    """
    Decide whether the service is in dry-run mode.

    TWILIO_DRY_RUN is read at call time. The function returns False ONLY when
    the stripped, lower-cased value is "false", "0", "no" or "off". Anything
    else (unset, empty, a typo) returns True, so a missing or mistyped value
    can never cause a real message to be sent.

    Returns:
        True if nothing should be sent to Twilio, False if live sending is
        explicitly enabled.
    """
    value = os.getenv("TWILIO_DRY_RUN", "").strip().lower()
    return value not in {"false", "0", "no", "off"}


def _normalize_e164(raw: str) -> str:
    """
    Convert a phone number to strict E.164 format, or reject it.

    Removes spaces, hyphens, dots and parentheses, and turns a leading "00"
    into "+". This is formatting only: the function never guesses a country
    code, so a national-format number such as "415-555-0192" is rejected.

    Parameters:
        raw: The phone number as supplied by the lead.

    Returns:
        The number in E.164 format, for example "+14155550192".

    Raises:
        ValueError: If the cleaned value is not valid E.164.
    """
    cleaned = re.sub(r"[\s\-.()]", "", raw)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    if not _E164_PATTERN.match(cleaned):
        raise ValueError(
            "Phone number must be in international format, for example "
            f"+14155550192 (got '{raw}')."
        )
    return cleaned


def _sanitize_body(text: str) -> str:
    """
    Replace typographic characters with GSM-7-safe equivalents.

    Curly quotes, dashes, ellipses and special spaces are mapped to ASCII,
    zero-width characters are deleted, and the result is stripped. One
    non-GSM character forces UCS-2 encoding, which cuts the per-segment
    limit from 160 to 70 characters, so this step directly controls cost.

    Parameters:
        text: The SMS body, as produced by the AI service.

    Returns:
        The sanitised body.

    Raises:
        ValueError: If nothing remains after sanitising.
    """
    cleaned = text.translate(_PUNCTUATION_MAP).strip()
    if not cleaned:
        raise ValueError("SMS body is empty after sanitising.")
    return cleaned


def _analyze_body(body: str) -> tuple[str, int, int]:
    """
    Detect the SMS encoding and estimate the segment count.

    If every character is in the GSM-7 basic set or extension table, the
    encoding is GSM-7: extension characters count as two septets, a single
    segment holds 160 septets, and concatenated segments hold 153 each.
    Otherwise the encoding is UCS-2: a single segment holds 70 UTF-16 code
    units, and concatenated segments hold 67 each.

    This is an estimate. The segment count shown in the Twilio Console logs
    is authoritative.

    Parameters:
        body: The sanitised SMS body.

    Returns:
        A tuple (encoding, units, segments).
    """
    if all(ch in _GSM7_BASIC or ch in _GSM7_EXTENSION for ch in body):
        units = sum(2 if ch in _GSM7_EXTENSION else 1 for ch in body)
        segments = 1 if units <= 160 else math.ceil(units / 153)
        return "GSM-7", units, segments

    units = len(body.encode("utf-16-le")) // 2
    segments = 1 if units <= 70 else math.ceil(units / 67)
    return "UCS-2", units, segments


def _sender_params() -> dict:
    """
    Build the sender argument for client.messages.create().

    If TWILIO_MESSAGING_SERVICE_SID is set (non-empty after stripping), it
    takes precedence and the result is {"messaging_service_sid": value}.
    Otherwise TWILIO_PHONE_NUMBER is required, normalised to E.164, and
    returned as {"from_": value}.

    Returns:
        A dict of keyword arguments for messages.create().

    Raises:
        SMSSendError: category "config", if TWILIO_PHONE_NUMBER is missing or
            not valid E.164.
    """
    service_sid = os.getenv("TWILIO_MESSAGING_SERVICE_SID", "").strip()
    if service_sid:
        return {"messaging_service_sid": service_sid}

    number = _require_env("TWILIO_PHONE_NUMBER")
    try:
        return {"from_": _normalize_e164(number)}
    except ValueError as exc:
        raise SMSSendError(
            "TWILIO_PHONE_NUMBER must be in E.164 format, e.g. +14155550192.",
            category="config",
        ) from exc


def _get_client() -> Client:
    """
    Build a Twilio client at call time.

    A new client is created on every call and never cached, so credential
    changes take effect without a restart. The HTTP client carries an
    explicit timeout because the SDK default is no timeout at all.

    Returns:
        A configured twilio.rest.Client.

    Raises:
        SMSSendError: category "config", if TWILIO_ACCOUNT_SID or
            TWILIO_AUTH_TOKEN is missing.
    """
    account_sid = _require_env("TWILIO_ACCOUNT_SID")
    auth_token = _require_env("TWILIO_AUTH_TOKEN")
    return Client(
        account_sid,
        auth_token,
        http_client=TwilioHttpClient(timeout=_REQUEST_TIMEOUT_SECONDS),
    )


def _classify_error(exc: Exception) -> SMSSendError:
    """
    Convert any exception raised while calling Twilio into an SMSSendError.

    Mapping for TwilioRestException (by Twilio error code):
        recipient: 21211, 21612, 21408, 21610, 21614
        transient: 20429, 21611, or any HTTP status of 500 and above
        config:    20003, 21212, 21606, and any unrecognised code

    A requests.exceptions.RequestException (network error or timeout) is
    transient. Any other exception is config, with only the exception type
    name in the message so that secrets are never echoed.

    Parameters:
        exc: The exception caught around client.messages.create().

    Returns:
        An SMSSendError ready to be raised by the caller.
    """
    if isinstance(exc, TwilioRestException):
        code = exc.code
        status = exc.status

        recipient_messages = {
            21211: "The destination number is not a valid phone number.",
            21612: "Twilio cannot route SMS to this destination number.",
            21408: (
                "SMS to this destination country is not enabled for the "
                "account. Enable it under Twilio Messaging Geo Permissions."
            ),
            21610: (
                "The recipient opted out (replied STOP). Do not message "
                "this number."
            ),
            21614: (
                "The destination number cannot receive SMS (not a mobile "
                "number)."
            ),
        }
        config_messages = {
            20003: (
                "Twilio authentication failed. Check TWILIO_ACCOUNT_SID and "
                "TWILIO_AUTH_TOKEN."
            ),
            21212: (
                "TWILIO_PHONE_NUMBER is invalid, not owned by this Twilio "
                "account, or not SMS-capable."
            ),
            21606: (
                "TWILIO_PHONE_NUMBER is invalid, not owned by this Twilio "
                "account, or not SMS-capable."
            ),
            572006: (
                "Trial accounts can only send predefined SMS templates. Upgrade the "
                "Twilio account to send custom message bodies."
            ),
            21608: (
                "The destination number is not a verified Caller ID. Trial accounts can "
                "only send SMS to verified numbers; add it under Phone Numbers > "
                "Verified Caller IDs, or upgrade the account."
            ),
        }

        if code in recipient_messages:
            category = "recipient"
            message = recipient_messages[code]
        elif code in (20429, 21611) or (status is not None and status >= 500):
            category = "transient"
            message = (
                "Twilio is rate limiting or temporarily unavailable. "
                "Retry later."
            )
        elif code in config_messages:
            category = "config"
            message = config_messages[code]
        else:
            category = "config"
            message = f"Twilio rejected the request: {exc.msg}"

        if code is not None:
            message += f" (Twilio error {code})"
        return SMSSendError(message, category=category, twilio_code=code)

    if isinstance(exc, requests.exceptions.RequestException):
        return SMSSendError(
            "Could not reach Twilio (network error or timeout).",
            category="transient",
        )

    return SMSSendError(
        f"Unexpected error while calling Twilio: {type(exc).__name__}.",
        category="config",
    )


# ---------------------------------------------------------------------------
# Public Function
# ---------------------------------------------------------------------------

def send_sms(payload: SMSPayload) -> dict:
    """
    Validate, prepare and (only when explicitly enabled) send one SMS.

    Order of operations. Steps 1 to 3 run identically in dry-run and live
    mode, so dry-run is a faithful rehearsal of validation:
        1. Normalise the recipient number to E.164.
        2. Sanitise the body to GSM-7-safe text.
        3. Detect the encoding and estimate the segment count.
        4. If dry-run is active, return a simulated result immediately,
           WITHOUT reading credentials, building a client or touching the
           network.
        5. Build the sender parameters and the Twilio client.
        6. Call client.messages.create(); any failure is classified.
        7. Return the live result.

    IMPORTANT: status "queued" means Twilio accepted the message. It does
    NOT mean the message was delivered; failures such as carrier blocking
    arrive later in Twilio's message logs. This function is also not
    idempotent: callers must check LeadRecord.sms_sent before calling it, and
    set it after success, or a retry can send a duplicate.

    Parameters:
        payload: A validated SMSPayload (to_phone, message, lead_score).

    Returns:
        A dict with keys: "message", "to", "message_id", "status",
        "dry_run", "encoding", "segments_estimated".

    Raises:
        ValueError: If the phone number is not valid E.164 or the body is
            empty after sanitising.
        SMSSendError: If configuration is missing or invalid (category
            "config"), or if Twilio rejects or fails the request (category
            "recipient", "config" or "transient").
    """
    # 1-3. Validation: identical in dry-run and live mode.
    to = _normalize_e164(payload.to_phone)
    body = _sanitize_body(payload.message)
    encoding, _units, segments = _analyze_body(body)

    # 4. Dry-run: return before any credential, client or network access.
    if _is_dry_run():
        return {
            "message": (
                "DRY RUN: SMS validated and prepared; nothing was sent. "
                "Set TWILIO_DRY_RUN=false to send real messages."
            ),
            "to": to,
            "message_id": None,
            "status": "dry_run",
            "dry_run": True,
            "encoding": encoding,
            "segments_estimated": segments,
        }

    # 5. Live mode: sender and client (both may raise SMSSendError "config").
    params = _sender_params()
    client = _get_client()

    # 6. Send, classifying every failure.
    try:
        msg = client.messages.create(to=to, body=body, **params)
    except SMSSendError:
        raise
    except Exception as exc:
        raise _classify_error(exc) from exc

    # 7. Accepted by Twilio (not necessarily delivered).
    return {
        "message": "SMS queued successfully",
        "to": to,
        "message_id": msg.sid,
        "status": msg.status,
        "dry_run": False,
        "encoding": encoding,
        "segments_estimated": segments,
    }