"""
services/calendly_service.py

Booking-link layer for the AI-powered real-estate lead system.

Responsibilities:
  - Build personalised, pre-filled Calendly scheduling URLs.
  - Inject those URLs into AI-written email and SMS copy deterministically
    (the LLM never handles a URL).
  - Degrade gracefully when no Calendly link is configured.

Design constraints:
  - Standard library only; no network calls.
  - Safe to import with an empty environment (no env-var reads at module level).
  - Pure ASCII source: non-ASCII literals are written as \\uXXXX escapes.
"""

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from models import LeadInput


# ---------------------------------------------------------------------------
# MODULE-LEVEL CONSTANTS  (no env reads here)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Token the LLM must embed in email copy to mark where the booking link goes.
LINK_TOKEN: str = "[BOOKING_LINK]"

# Appended to every outgoing SMS.
SMS_OPT_OUT: str = "Reply STOP to opt out"

# Hard character budget for a single SMS segment.
SMS_MAX_CHARS: int = 160

# Minimum text body characters required before the suffix to accept a plan.
_MIN_SMS_TEXT: int = 45

# UTM source injected into every tracked link.
_UTM_SOURCE: str = "lead_system"

# Default value for utm_campaign when the env var is absent or blank.
_DEFAULT_CAMPAIGN: str = "ai_followup"

# Maximum characters kept from a cleaned name component.
_MAX_NAME_CHARS: int = 100

# Maximum characters kept from a cleaned UTM value.
_MAX_UTM_CHARS: int = 50

# CTA sentence appended to email body when LINK_TOKEN is absent.
_EMAIL_CTA_FALLBACK: str = "Book a time that suits you:"

# Detects any HTTP/HTTPS/bare URL or Calendly domain reference in free text.
_URL_PATTERN: re.Pattern = re.compile(
    r"(?:https?://|www\.)\S+|\bcalendly\.com/\S*",
    re.IGNORECASE,
)

# Matches an opt-out instruction so it can be stripped before re-appending.
_OPT_OUT_PATTERN: re.Pattern = re.compile(
    r"\s*reply\s+stop\b[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# DATACLASS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmsPlan:
    """
    Immutable plan produced by plan_sms() and consumed by finish_sms().

    Attributes:
        link:        The booking URL to append, or None when no link is available.
        suffix:      The exact string to append after the text body, including
                     a leading space, the optional link, and the opt-out notice.
        text_budget: Maximum characters available for the main SMS text body.
    """

    link: str | None
    suffix: str
    text_budget: int


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------


def _normalize_base_link(raw: object) -> str | None:
    """
    Validate and normalise a raw Calendly URL string.

    Steps:
      1. Reject non-str input immediately.
      2. Strip outer whitespace and surrounding quote characters, then
         strip again; reject if blank or if the remaining value contains
         any whitespace (internal spaces are invalid in a URL).
      3. Prefix ``https://`` when no ``://`` is present.
      4. Parse with ``urlsplit``; reject on ValueError.
      5. Scheme must be ``http`` or ``https``.
      6. ``netloc`` must be non-empty, contain a dot, and not contain ``@``
         (credentials embedded in URLs are rejected).
      7. ``path`` stripped of ``/`` must be non-empty (bare domain not enough).
      8. Return the reconstructed URL with scheme forced to ``https``.

    Parameters:
        raw: Candidate value from the environment (any type accepted).

    Returns:
        A normalised ``https://`` URL string, or ``None`` if validation fails.

    Raises:
        Nothing -- all error paths return ``None``.
    """
    if not isinstance(raw, str):
        return None

    value = raw.strip().strip("\"'").strip()

    if not value:
        return None

    # Internal whitespace is never valid in a URL.
    if re.search(r"\s", value):
        return None

    # Ensure a scheme is present so urlsplit parses netloc correctly.
    if "://" not in value:
        value = "https://" + value

    try:
        parts = urlsplit(value)
    except ValueError:
        return None

    if parts.scheme not in ("http", "https"):
        return None

    netloc = parts.netloc
    if not netloc:
        return None
    if "." not in netloc:
        return None
    if "@" in netloc:
        return None

    if not parts.path.strip("/"):
        return None

    return urlunsplit(("https", netloc, parts.path, parts.query, parts.fragment))


def _clean_name(value: object) -> str:
    """
    Sanitise a name string for use in a Calendly pre-fill parameter.

    Steps:
      1. Return ``""`` for non-str input.
      2. Collapse all runs of whitespace to single spaces and strip.
      3. Remove any character whose Unicode general category starts with
         ``"C"`` (control, format, surrogate, private-use, unassigned).
      4. Cap the result at ``_MAX_NAME_CHARS`` characters.

    Parameters:
        value: Raw name value (any type accepted).

    Returns:
        A clean, URL-safe name string (may be empty).

    Raises:
        Nothing.
    """
    if not isinstance(value, str):
        return ""
    normalised = " ".join(value.split())
    cleaned = "".join(
        ch for ch in normalised
        if not unicodedata.category(ch).startswith("C")
    )
    return cleaned[:_MAX_NAME_CHARS]


def _split_name(name: str) -> tuple[str, str]:
    """
    Split a full name string into first and last components.

    The first token becomes the first name; all remaining tokens joined by a
    space become the last name.  An empty input produces two empty strings.

    Parameters:
        name: A cleaned full name string.

    Returns:
        A ``(first_name, last_name)`` tuple; both elements are strings and
        either may be empty.

    Raises:
        Nothing.
    """
    parts = name.split()
    if not parts:
        return ("", "")
    return (parts[0], " ".join(parts[1:]))


def _clean_utm(value: object, default: str) -> str:
    """
    Sanitise a string for use as a UTM parameter value.

    Steps:
      1. Keep only characters in ``[A-Za-z0-9_.-]``; replace every run of
         disallowed characters with a single underscore.
      2. Strip leading and trailing underscores.
      3. Cap at ``_MAX_UTM_CHARS`` characters.
      4. If the result is empty, return ``default``.

    Parameters:
        value:   Raw UTM value (any type accepted).
        default: Fallback returned when the cleaned result is empty.

    Returns:
        A sanitised UTM value string, never empty.

    Raises:
        Nothing.
    """
    if not isinstance(value, str):
        return default
    cleaned = re.sub(r"[^A-Za-z0-9_.\\-]+", "_", value).strip("_")
    cleaned = cleaned[:_MAX_UTM_CHARS]
    return cleaned if cleaned else default


def _name_mode() -> str:
    """
    Return the configured name-field mode for Calendly pre-fill.

    Reads ``CALENDLY_NAME_FIELDS`` from the environment at call time.
    Returns ``"split"`` only when the variable (stripped, lower-cased) equals
    exactly ``"split"``; otherwise returns ``"single"``.

    Parameters:
        None.

    Returns:
        ``"split"`` or ``"single"``.

    Raises:
        Nothing.
    """
    raw = os.getenv("CALENDLY_NAME_FIELDS", "")
    return "split" if raw.strip().lower() == "split" else "single"


def _compose_url(
    base: str,
    lead: object,
    channel: str,
    include_utm: bool,
) -> str:
    """
    Compose a final Calendly scheduling URL from a validated base link.

    Pre-fill parameters (email channel only):
      - Existing query pairs whose keys overlap with the keys being added are
        removed first so they can be replaced.
      - In ``"single"`` name mode: appends ``name`` (cleaned full name).
      - In ``"split"`` name mode: appends ``first_name`` and ``last_name``
        separately; a component that is empty is omitted entirely.
      - Appends ``email`` (stripped, must contain ``"@"``, at most 254 chars).
      - Any value that is empty after cleaning is skipped.

    UTM parameters (when ``include_utm`` is True):
      - ``email`` channel: ``utm_source``, ``utm_medium=email``,
        ``utm_campaign`` (cleaned from ``CALENDLY_UTM_CAMPAIGN`` env var).
      - ``sms`` channel: ``utm_medium=sms`` only.
      - A UTM key already present in the base URL query string is left
        untouched (the operator-configured value takes precedence).

    Encoding: ``urllib.parse.quote`` is used so spaces become ``%20``,
    ``@`` becomes ``%40``, and ``+`` becomes ``%2B``.

    Parameters:
        base:        A normalised ``https://`` Calendly URL (from
                     ``_normalize_base_link``).
        lead:        Any object; ``name`` and ``email`` are accessed via
                     ``getattr`` with safe defaults so incomplete objects
                     never raise.
        channel:     ``"email"`` or ``"sms"``.
        include_utm: When ``True``, UTM parameters are appended (subject to
                     the existing-key rule above).

    Returns:
        The fully composed URL string.

    Raises:
        Nothing.
    """
    parsed = urlsplit(base)
    pairs: list[tuple[str, str]] = parse_qsl(
        parsed.query, keep_blank_values=True
    )

    # --- Pre-fill (email only) ---
    if channel == "email":
        mode = _name_mode()
        raw_name = getattr(lead, "name", "") or ""
        cleaned_name = _clean_name(raw_name)

        if mode == "single":
            keys_to_remove = {"name", "email"}
            pairs = [(k, v) for k, v in pairs if k not in keys_to_remove]
            if cleaned_name:
                pairs.append(("name", cleaned_name))
        else:
            first, last = _split_name(cleaned_name)
            keys_to_remove = {"first_name", "last_name", "email"}
            pairs = [(k, v) for k, v in pairs if k not in keys_to_remove]
            if first:
                pairs.append(("first_name", first))
            if last:
                pairs.append(("last_name", last))

        raw_email = getattr(lead, "email", "") or ""
        email_str = str(raw_email).strip()
        if "@" in email_str and len(email_str) <= 254:
            pairs.append(("email", email_str))

    # --- UTM parameters ---
    if include_utm:
        existing_utm_keys = {k for k, _ in pairs}
        if channel == "email":
            utm_additions: list[tuple[str, str]] = [
                ("utm_source", _UTM_SOURCE),
                ("utm_medium", "email"),
                (
                    "utm_campaign",
                    _clean_utm(
                        os.getenv("CALENDLY_UTM_CAMPAIGN", ""),
                        _DEFAULT_CAMPAIGN,
                    ),
                ),
            ]
            for key, val in utm_additions:
                if key not in existing_utm_keys:
                    pairs.append((key, val))
        elif channel == "sms":
            if "utm_medium" not in existing_utm_keys:
                pairs.append(("utm_medium", "sms"))

    new_query = urlencode(pairs, quote_via=quote)
    return urlunsplit((
        "https",
        parsed.netloc,
        parsed.path,
        new_query,
        parsed.fragment,
    ))


# ---------------------------------------------------------------------------
# PUBLIC FUNCTIONS
# ---------------------------------------------------------------------------


def get_base_link() -> str | None:
    """
    Retrieve and validate the configured Calendly base scheduling URL.

    Reads ``CALENDLY_LINK`` from the environment at call time.  If the
    variable is unset or blank, returns ``None`` silently.  If the variable
    is set but the value fails validation, logs exactly one ``WARNING`` through
    this module's logger (the message never includes the variable's value or
    any lead data) and returns ``None``.

    Parameters:
        None.

    Returns:
        A normalised ``https://`` Calendly URL string, or ``None``.

    Raises:
        Nothing.
    """
    raw = os.getenv("CALENDLY_LINK")
    if not raw or not raw.strip():
        return None
    result = _normalize_base_link(raw)
    if result is None:
        logger.warning(
            "CALENDLY_LINK is set but is not a valid scheduling URL; "
            "booking links are disabled."
        )
    return result


def build_booking_link(
    lead: LeadInput,
    *,
    channel: str,
    include_utm: bool = True,
) -> str | None:
    """
    Build a personalised Calendly booking URL for a given lead and channel.

    The URL is composed deterministically from the validated base link plus
    pre-fill and UTM parameters appropriate to the channel.  The LLM is
    never involved in URL construction.

    Parameters:
        lead:        A ``LeadInput`` instance (or any object with ``name``
                     and ``email`` attributes; missing attributes are handled
                     gracefully).
        channel:     Delivery channel; must be ``"email"`` or ``"sms"``.
                     Email links include name/email pre-fill and full UTM tags.
                     SMS links carry no personal data and at most
                     ``utm_medium=sms``.
        include_utm: When ``True`` (default), UTM parameters are appended.

    Returns:
        A fully composed Calendly URL string, or ``None`` when no base link
        is configured.

    Raises:
        ValueError: When ``channel`` is neither ``"email"`` nor ``"sms"``.
    """
    if channel not in ("email", "sms"):
        raise ValueError(
            f"channel must be 'email' or 'sms', got {channel!r}"
        )
    base = get_base_link()
    if base is None:
        return None
    raw_name = getattr(lead, "name", "")
    raw_email = getattr(lead, "email", "")
    # Create a lightweight namespace so _compose_url can use getattr safely.
    return _compose_url(base, lead, channel, include_utm)


def strip_urls(text: object) -> str:
    """
    Remove all URL-like patterns from a string.

    Uses ``_URL_PATTERN`` to match HTTP/HTTPS URLs, ``www.`` references, and
    bare ``calendly.com/`` paths.  After removal, runs of spaces or tabs are
    collapsed to a single space and spaces immediately before a newline are
    removed.  Newlines themselves are preserved.

    Parameters:
        text: Input value (any type accepted; non-str returns ``""``).

    Returns:
        The cleaned string with URLs removed and whitespace normalised.

    Raises:
        Nothing.
    """
    if not isinstance(text, str):
        return ""
    removed = _URL_PATTERN.sub("", text)
    # Collapse horizontal whitespace, but preserve newlines.
    collapsed = re.sub(r"[ \t]+", " ", removed)
    # Drop spaces that sit immediately before a newline.
    no_trailing_space = re.sub(r" +\n", "\n", collapsed)
    return no_trailing_space


def inject_link_into_email_body(body: object, link: str | None) -> str:
    """
    Insert a booking URL into an AI-written email body deterministically.

    Processing pipeline:
      1. ``strip_urls`` is applied to ``body`` (non-str becomes ``""``);
         ``\\r\\n`` sequences are normalised to ``\\n``.
      2. If ``link`` is provided:
           - The first occurrence of ``LINK_TOKEN`` is replaced with
             ``"\\n" + link + "\\n"``; all subsequent tokens are deleted.
           - If ``LINK_TOKEN`` is absent, ``"\\n\\n" + _EMAIL_CTA_FALLBACK +
             "\\n" + link`` is appended.
         If ``link`` is ``None`` or empty: every ``LINK_TOKEN`` is deleted.
      3. Trailing spaces on every line are stripped; runs of three or more
         consecutive newlines are collapsed to two; the entire result is
         stripped of leading/trailing whitespace.

    The booking URL always sits alone on its own line in the finished body.

    Parameters:
        body: AI-generated email body (any type; non-str treated as ``""``).
        link: Pre-built Calendly URL from ``build_booking_link``, or ``None``.

    Returns:
        The cleaned, link-injected email body string.

    Raises:
        Nothing.
    """
    cleaned = strip_urls(body).replace("\r\n", "\n")

    if link:
        if LINK_TOKEN in cleaned:
            # Replace first token, delete the rest.
            first, *rest = cleaned.split(LINK_TOKEN)
            remainder = "".join(rest)
            cleaned = first + "\n" + link + "\n" + remainder.replace(LINK_TOKEN, "")
        else:
            cleaned = cleaned + "\n\n" + _EMAIL_CTA_FALLBACK + "\n" + link
    else:
        cleaned = cleaned.replace(LINK_TOKEN, "")

    # Strip trailing spaces on each line.
    lines = [line.rstrip(" ") for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)

    # Collapse 3+ consecutive newlines to 2.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def plan_sms(lead: LeadInput) -> SmsPlan:
    """
    Determine the best SMS suffix plan for a given lead.

    Candidates are evaluated in order:
      1. SMS booking link with UTM tagging (``utm_medium=sms``).
      2. SMS booking link without UTM tagging.
    Duplicate candidate URLs and ``None`` values are skipped.

    For each candidate, the suffix is ``" " + link + " " + SMS_OPT_OUT`` and
    the text budget is ``SMS_MAX_CHARS - len(suffix)``.  The first candidate
    whose budget is at least ``_MIN_SMS_TEXT`` characters is used.

    If no candidate meets the budget, the plan has ``link=None`` and a suffix
    of ``" " + SMS_OPT_OUT``.

    Parameters:
        lead: A ``LeadInput`` instance (used only to call ``build_booking_link``
              which applies privacy rules; no personal data appears in SMS URLs).

    Returns:
        A frozen ``SmsPlan`` dataclass instance.

    Raises:
        Nothing.
    """
    base = get_base_link()

    # Build candidate links once (sms channel carries no personal data).
    seen: set[str] = set()
    candidates: list[str] = []
    for include_utm in (True, False):
        if base is None:
            link_candidate = None
        else:
            link_candidate = _compose_url(base, lead, "sms", include_utm)
        if link_candidate is not None and link_candidate not in seen:
            seen.add(link_candidate)
            candidates.append(link_candidate)

    for link in candidates:
        suffix = " " + link + " " + SMS_OPT_OUT
        budget = SMS_MAX_CHARS - len(suffix)
        if budget >= _MIN_SMS_TEXT:
            return SmsPlan(link=link, suffix=suffix, text_budget=budget)

    # No link fits; opt-out only.
    fallback_suffix = " " + SMS_OPT_OUT
    return SmsPlan(
        link=None,
        suffix=fallback_suffix,
        text_budget=SMS_MAX_CHARS - len(fallback_suffix),
    )


def finish_sms(text: object, plan: SmsPlan, lead: LeadInput) -> str:
    """
    Finalise an AI-written SMS body by cleaning, truncating, and appending the
    booking link and opt-out notice from an ``SmsPlan``.

    Processing pipeline:
      1. ``strip_urls`` is applied to ``text`` (non-str becomes ``""``);
         ``_OPT_OUT_PATTERN`` matches are removed; the Unicode ellipsis
         character (U+2026) is replaced with three ASCII full stops;
         all whitespace is collapsed to single spaces and the result is
         stripped.
      2. If ``len(text) > plan.text_budget``: the text is cut to
         ``text_budget`` characters.  If the last space in the truncated
         portion falls at or after ``text_budget // 2``, the cut backs off
         to that space to avoid splitting mid-word.  Leading/trailing
         ``" ,;:-"`` are stripped from the result.
      3. If the result contains no ASCII letter or digit, a safe fallback
         message is used: ``"Hi <first>, an agent will contact you shortly."``
         where ``<first>`` is derived from ``lead.name`` (only
         letters, digits, apostrophes, and hyphens; at most 30 chars).  When
         no usable first name can be derived, ``"Hi, an agent will contact
         you shortly."`` is used instead.  The fallback is also truncated to
         the budget using the same rule as step 2.
      4. ``message = text + plan.suffix`` is assembled.  A ``ValueError`` is
         raised if ``len(message) > SMS_MAX_CHARS`` (this must never happen
         given correct plan construction).

    The ``plan.suffix`` (link and opt-out) is never truncated or altered.

    Parameters:
        text: AI-generated SMS body copy (any type; non-str treated as ``""``).
        plan: An ``SmsPlan`` produced by ``plan_sms`` for this lead.
        lead: The ``LeadInput`` for the lead being messaged (used to derive
              a safe first name for the fallback message).

    Returns:
        The final SMS message string, guaranteed to be at most
        ``SMS_MAX_CHARS`` characters long.

    Raises:
        ValueError: When the assembled message exceeds ``SMS_MAX_CHARS``
                    (indicates a logic error in plan construction).
    """
    # Step 1: clean the text body.
    cleaned = strip_urls(text)
    cleaned = _OPT_OUT_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("\u2026", "...")
    cleaned = " ".join(cleaned.split())

    # Step 2: truncate to text_budget if necessary.
    def _truncate(s: str, budget: int) -> str:
        if len(s) <= budget:
            return s
        truncated = s[:budget]
        last_space = truncated.rfind(" ")
        if last_space >= budget // 2:
            truncated = truncated[:last_space]
        return truncated.rstrip(" ,;:-")

    cleaned = _truncate(cleaned, plan.text_budget)

    # Step 3: fallback when no alphanumeric content remains.
    if not re.search(r"[A-Za-z0-9]", cleaned):
        raw_lead_name = getattr(lead, "name", "") or ""
        first_raw = _clean_name(raw_lead_name).split()[0] if _clean_name(raw_lead_name).split() else ""
        # Keep only letters, digits, apostrophes, hyphens; cap at 30.
        first_safe = re.sub(r"[^A-Za-z0-9'\-]", "", first_raw)[:30]
        if first_safe:
            fallback = f"Hi {first_safe}, an agent will contact you shortly."
        else:
            fallback = "Hi, an agent will contact you shortly."
        cleaned = _truncate(fallback, plan.text_budget)

    # Step 4: assemble and validate.
    message = cleaned + plan.suffix
    if len(message) > SMS_MAX_CHARS:
        raise ValueError(
            f"Assembled SMS message is {len(message)} characters, "
            f"which exceeds the {SMS_MAX_CHARS}-character limit. "
            "This indicates a logic error in SmsPlan construction."
        )
    return message
