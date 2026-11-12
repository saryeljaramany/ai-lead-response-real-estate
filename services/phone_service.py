"""
services/phone_service.py

Phone number normalisation service for the AI-powered lead management system.

Provides pure, deterministic E.164 phone number validation and normalisation.
It operates without network access and is completely safe to import when no
environment variables are configured.

Contract:
    PhoneNormalization: Immutable result container (e164, reason).
    normalize_phone(raw): Normalises phone input to E.164 without raising.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dataclasses import dataclass
import logging
import os
import re

import phonenumbers
from phonenumbers import PhoneNumberFormat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# E.164: a plus sign, a non-zero first digit, 8 to 15 digits in total.
_E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")

# Separators stripped during normalisation (spaces, hyphens, dots, parens).
_SEPARATORS = re.compile(r"[\s\-.()]")

# Module logger.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhoneNormalization:
    """
    Immutable container representing the outcome of phone normalisation.

    Attributes:
        e164: The normalised phone number in strict E.164 format
            (e.g. '+14155550192'), or None if normalisation failed.
        reason: Category describing the outcome. Exactly one of:
            - 'international': Input was supplied with an international prefix
              (+ or 00) and satisfies E.164 syntax.
            - 'national': Input had no country code but was successfully
              normalised to E.164 using the configured DEFAULT_COUNTRY_CODE.
            - 'needs_country_code': Input was national digits but no
              valid DEFAULT_COUNTRY_CODE was configured.
            - 'invalid': Input was not a string, was empty, contained
              invalid characters, or could not be parsed.
            - 'invalid_for_region': Input was parsed using the default region
              but is not a valid number for that region.
    """

    e164: str | None
    reason: str


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _default_region() -> str | None:
    """
    Resolve the default ISO 3166-1 alpha-2 region code from the environment.

    Reads DEFAULT_COUNTRY_CODE inside the function at call time. Accepts
    either a two-letter ISO country code (e.g. 'US', 'GB', 'IT') or a numeric
    international calling code with optional leading '+' or '00' (e.g. '1',
    '+1', '0039'). Surrounding whitespace and quotes are stripped.

    Returns:
        The uppercase two-letter region code if valid and supported by
        libphonenumber, or None if DEFAULT_COUNTRY_CODE is unset, empty,
        or invalid.

    Raises:
        None. All parsing failures are handled internally without raising.
    """
    raw = os.getenv("DEFAULT_COUNTRY_CODE")
    if raw is None or not raw.strip():
        return None

    value = raw.strip().strip("\"'").strip()

    if len(value) == 2 and value.isalpha():
        region = value.upper()
        if region in phonenumbers.SUPPORTED_REGIONS:
            return region
    else:
        digits = value
        if digits.startswith("+"):
            digits = digits[1:]
        elif digits.startswith("00"):
            digits = digits[2:]

        if digits.isascii() and digits.isdigit():
            region = phonenumbers.region_code_for_country_code(int(digits))
            if region and region != "ZZ":
                return region

    logger.warning(
        "DEFAULT_COUNTRY_CODE is set but is not a valid ISO country code or calling code; "
        "national phone numbers will be rejected."
    )
    return None


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def normalize_phone(raw: object) -> PhoneNormalization:
    """
    Normalise a raw phone number into strict E.164 format.

    Guaranteed never to raise an exception under any circumstance. Handles
    both international numbers (with '+' or '00' prefixes) and national
    numbers resolved against the default region from DEFAULT_COUNTRY_CODE.

    Evaluation steps:
        1. Non-string or blank input -> (None, 'invalid').
        2. Strip whitespace and separators [ -.(]. Replace leading '00' with '+'.
        3. If starts with '+': validate against E.164 pattern. Valid -> (cleaned, 'international');
           invalid -> (None, 'invalid'). Libphonenumber validity is bypassed to allow
           Twilio magic numbers (e.g. +15005550006).
        4. If not all digits -> (None, 'invalid').
        5. Read default region. If None -> (None, 'needs_country_code'). Never guess.
        6. Parse with phonenumbers. If parse fails -> (None, 'invalid').
           If invalid for region -> (None, 'invalid_for_region').
           Otherwise format to E.164 -> (formatted, 'national').

    Parameters:
        raw: The raw input to normalise (expected str, but accepts any type).

    Returns:
        PhoneNormalization: An immutable object with fields .e164 (str | None)
        and .reason (str).

    Raises:
        None. This function is guaranteed never to raise.
    """
    # 1. Not a str, or blank -> (None, "invalid").
    if not isinstance(raw, str):
        return PhoneNormalization(e164=None, reason="invalid")
    stripped = raw.strip()
    if not stripped:
        return PhoneNormalization(e164=None, reason="invalid")

    # 2. cleaned = _SEPARATORS.sub("", raw.strip()); a leading "00" becomes "+".
    cleaned = _SEPARATORS.sub("", stripped)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]

    # 3. Starts with "+": (cleaned, "international") when it matches _E164_PATTERN,
    #    otherwise (None, "invalid"). Do NOT use libphonenumber validity here, so
    #    Twilio's magic numbers such as +15005550006 still pass (same rule as Phase 5).
    if cleaned.startswith("+"):
        if _E164_PATTERN.match(cleaned):
            return PhoneNormalization(e164=cleaned, reason="international")
        return PhoneNormalization(e164=None, reason="invalid")

    # 4. Not all digits -> (None, "invalid").
    if not cleaned.isdigit():
        return PhoneNormalization(e164=None, reason="invalid")

    # 5. region = _default_region(); None -> (None, "needs_country_code"). Never guess.
    region = _default_region()
    if region is None:
        return PhoneNormalization(e164=None, reason="needs_country_code")

    # 6. try phonenumbers.parse(cleaned, region); NumberParseException -> (None,
    #    "invalid"). If not phonenumbers.is_valid_number(number) -> (None,
    #    "invalid_for_region"). Otherwise (phonenumbers.format_number(number,
    #    phonenumbers.PhoneNumberFormat.E164), "national").
    try:
        number = phonenumbers.parse(cleaned, region)
    except Exception:
        return PhoneNormalization(e164=None, reason="invalid")

    try:
        if not phonenumbers.is_valid_number(number):
            return PhoneNormalization(e164=None, reason="invalid_for_region")

        formatted = phonenumbers.format_number(
            number,
            PhoneNumberFormat.E164,
        )
        return PhoneNormalization(e164=formatted, reason="national")
    except Exception:
        return PhoneNormalization(e164=None, reason="invalid")
