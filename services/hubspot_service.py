"""
services/hubspot_service.py

HubSpot CRM integration layer for the AI-powered real-estate lead system.

This module provides a single idempotent upsert operation that creates or
updates a HubSpot contact keyed on email address.  It is designed so that:

    - Importing the module with an empty environment never raises.
    - Environment variables are read only inside functions, after load_dotenv()
      has run in main.py.
    - No HTTP request is made unless live mode is explicitly enabled by
      setting HUBSPOT_DRY_RUN=false.
    - Every failure is classified into a category (config, invalid, conflict,
      transient) so that the HTTP layer and n8n pipeline can decide whether
      a retry makes sense.

Privacy: contact PII (email, name, phone) and the HubSpot access token are
never included in exception messages.  Only fixed diagnostic text, the HTTP
status code, the HubSpot error category, and the HubSpot correlation ID are
surfaced.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from models import HubSpotContact, LeadRecord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL: str = "https://api.hubapi.com"
_API_VERSION: str = "2026-09"
_UPSERT_PATH: str = "/crm/objects/{version}/contacts/batch/upsert"
_REQUEST_TIMEOUT_SECONDS: int = 10
_MISSING_LASTNAME: str = "-"

# HubSpot DEFAULT contact properties (exist in every account).
_STANDARD_PROPERTIES: tuple[str, ...] = ("firstname", "lastname", "phone")

# CUSTOM contact properties, created by hand in HubSpot.
# Mapping: model field name -> HubSpot internal property name.
_CUSTOM_PROPERTIES: dict[str, str] = {
    "lead_score": "lead_score",
    "lead_status": "lead_status",
    "property_type": "property_type",
    "preferred_location": "preferred_location",
}

# email is neither standard nor custom here: it is the record identifier
# (id + idProperty) and must NOT appear in the properties dict.


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class CRMError(RuntimeError):
    """Classified CRM integration error.

    Every instance carries a ``category`` that tells callers whether a retry
    is reasonable:

        config     -- token, scope, endpoint, or HubSpot-account setup
                      problem; retrying will not help.
        invalid    -- HubSpot rejected the contact data itself; do not retry.
        conflict   -- a persistent conflicting contact exists; do not retry.
        transient  -- HubSpot or the network failed; a retry is reasonable.

    Parameters
    ----------
    message : str
        A fixed, PII-free diagnostic string defined in this module.
    category : str
        One of ``"config"``, ``"invalid"``, ``"conflict"``, ``"transient"``.
    status : int or None
        The HTTP status code returned by HubSpot, if available.
    correlation_id : str or None
        The ``correlationId`` returned by HubSpot, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        status: int | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category: str = category
        self.status: int | None = status
        self.correlation_id: str | None = correlation_id


# ---------------------------------------------------------------------------
# Private helpers -- environment
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
    CRMError
        With ``category="config"`` if the variable is unset or blank.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        raise CRMError(
            f"{name} is missing or empty. Set it in .env and restart the server.",
            category="config",
        )
    return value.strip()


def _is_dry_run() -> bool:
    """Determine whether the service should operate in dry-run mode.

    Reads ``HUBSPOT_DRY_RUN`` at call time.  Returns ``False`` **only** when
    the stripped, lower-cased value is one of ``"false"``, ``"0"``, ``"no"``,
    or ``"off"``.  Any other value -- including unset, empty, or a typo --
    returns ``True`` so that a missing or mistyped flag never writes to
    HubSpot.

    Returns
    -------
    bool
        ``True`` if the upsert should be skipped (dry-run mode);
        ``False`` if the upsert should be executed against HubSpot.
    """
    raw = os.getenv("HUBSPOT_DRY_RUN")
    if raw is None:
        return True
    return raw.strip().lower() not in ("false", "0", "no", "off")


# ---------------------------------------------------------------------------
# Private helpers -- name splitting
# ---------------------------------------------------------------------------

def _split_name(full_name: str) -> tuple[str, str]:
    """Split a full name into first and last name parts.

    The first whitespace-delimited token becomes the first name.  All
    remaining tokens are joined with a single space to form the last name.
    If only one token is present, the last name defaults to
    ``_MISSING_LASTNAME`` (``"-"``).

    Examples::

        "Maria de la Cruz"  -> ("Maria", "de la Cruz")
        "Sarah"             -> ("Sarah", "-")

    Parameters
    ----------
    full_name : str
        The full name string to split.

    Returns
    -------
    tuple[str, str]
        A ``(firstname, lastname)`` pair.
    """
    parts = full_name.split()
    first = parts[0]
    last = " ".join(parts[1:]) or _MISSING_LASTNAME
    return first, last


# ---------------------------------------------------------------------------
# Public helper -- model conversion
# ---------------------------------------------------------------------------

def contact_from_record(record: LeadRecord) -> HubSpotContact:
    """Build a ``HubSpotContact`` from a scored ``LeadRecord``.

    This is a pure function with no side effects: it reads fields from
    ``record.lead`` and ``record.score`` and returns a fully validated
    ``HubSpotContact`` instance.  It is used by the Phase 9 pipeline to
    bridge the lead model into the CRM layer.

    Parameters
    ----------
    record : LeadRecord
        A scored lead record containing ``lead`` (``LeadInput``) and
        ``score`` (``LeadScore``) sub-models.

    Returns
    -------
    HubSpotContact
        A validated HubSpot contact ready for ``upsert_contact``.
    """
    firstname, lastname = _split_name(record.lead.name)
    return HubSpotContact(
        email=record.lead.email,
        firstname=firstname,
        lastname=lastname,
        phone=record.lead.phone,
        property_type=record.lead.property_type,
        preferred_location=record.lead.preferred_location,
        lead_score=record.score.score,
        lead_status=record.score.category,
    )


# ---------------------------------------------------------------------------
# Private helpers -- payload construction
# ---------------------------------------------------------------------------

def _build_properties(contact: HubSpotContact) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build the HubSpot properties dict from a validated contact.

    Standard properties (``firstname``, ``lastname``, ``phone``) are included
    only when their stripped value is non-empty.  Custom properties are
    included only when their value is not ``None`` and not blank after
    stripping.  ``lead_score`` is converted to ``str(int)``.

    The ``email`` field is **never** included in properties -- it travels
    as the ``id`` + ``idProperty`` pair in the upsert payload.

    Empty strings are never sent because an empty string **clears** a
    property in HubSpot.

    Parameters
    ----------
    contact : HubSpotContact
        A validated HubSpot contact.

    Returns
    -------
    tuple[dict[str, str], dict[str, list[str]]]
        A ``(properties, summary)`` pair.  ``properties`` maps HubSpot
        internal property names to string values.  ``summary`` has keys
        ``"standard"`` and ``"custom"``, each listing the HubSpot internal
        names that were included (never values).
    """
    properties: dict[str, str] = {}
    standard_names: list[str] = []
    custom_names: list[str] = []

    # -- Standard properties -------------------------------------------------
    for prop_name in _STANDARD_PROPERTIES:
        value = getattr(contact, prop_name, None)
        if value is not None and str(value).strip():
            properties[prop_name] = str(value).strip()
            standard_names.append(prop_name)

    # -- Custom properties ---------------------------------------------------
    for field_name, hubspot_name in _CUSTOM_PROPERTIES.items():
        value = getattr(contact, field_name, None)
        if value is None:
            continue
        if field_name == "lead_score":
            str_value = str(int(value))
        else:
            str_value = str(value).strip()
        if str_value:
            properties[hubspot_name] = str_value
            custom_names.append(hubspot_name)

    summary: dict[str, list[str]] = {
        "standard": standard_names,
        "custom": custom_names,
    }
    return properties, summary


def _build_payload(
    contact: HubSpotContact,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Build the full HubSpot batch-upsert request payload.

    Wraps the output of ``_build_properties`` into the structure expected by
    ``POST /crm/objects/{version}/contacts/batch/upsert``.

    Parameters
    ----------
    contact : HubSpotContact
        A validated HubSpot contact.

    Returns
    -------
    tuple[dict, dict]
        A ``(payload, summary)`` pair.  ``payload`` is the JSON-serialisable
        request body.  ``summary`` lists the property names that were
        included (never values).
    """
    properties, summary = _build_properties(contact)
    payload: dict[str, Any] = {
        "inputs": [
            {
                "idProperty": "email",
                "id": str(contact.email).strip().lower(),
                "properties": properties,
            }
        ]
    }
    return payload, summary


# ---------------------------------------------------------------------------
# Private helpers -- HTTP client
# ---------------------------------------------------------------------------

def _get_http_client() -> httpx.Client:
    """Create a per-call ``httpx.Client`` configured for the HubSpot API.

    Reads ``HUBSPOT_ACCESS_TOKEN`` via ``_require_env`` at call time.  The
    client must be used as a context manager and is never cached.

    Returns
    -------
    httpx.Client
        A configured HTTP client with base URL, timeout, and authorization
        header set.

    Raises
    ------
    CRMError
        With ``category="config"`` if the access token is missing.
    """
    token = _require_env("HUBSPOT_ACCESS_TOKEN")
    return httpx.Client(
        base_url=_BASE_URL,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )


# ---------------------------------------------------------------------------
# Private helpers -- response handling
# ---------------------------------------------------------------------------

def _safe_json(response: httpx.Response) -> dict[str, Any] | None:
    """Attempt to parse the response body as JSON.

    Returns the parsed dictionary if the body is valid JSON and decodes to a
    ``dict``.  Returns ``None`` otherwise (a non-JSON body is normal for
    gateway errors).

    Parameters
    ----------
    response : httpx.Response
        The HTTP response to parse.

    Returns
    -------
    dict or None
        The parsed JSON body, or ``None`` if parsing fails or the result is
        not a dict.
    """
    try:
        data = response.json()
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _collect_strings(obj: Any) -> list[str]:
    """Recursively collect every string value from a nested structure.

    Parameters
    ----------
    obj : Any
        A JSON-decoded object (dict, list, str, number, bool, None).

    Returns
    -------
    list[str]
        All string values found, upper-cased for keyword matching.
    """
    results: list[str] = []
    if isinstance(obj, str):
        results.append(obj.upper())
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_collect_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_strings(item))
    return results


def _classify_failure(status: int, body: dict[str, Any] | None) -> CRMError:
    """Classify an HTTP failure into a ``CRMError`` with the correct category.

    Inspects the HTTP status code and the response body to produce a
    fixed-text diagnostic message.  **No PII or response-body text** is
    included in the message -- only the HTTP status, the HubSpot error
    category string, and the correlation ID.

    Parameters
    ----------
    status : int
        The HTTP status code.
    body : dict or None
        The parsed JSON response body, or ``None``.

    Returns
    -------
    CRMError
        A classified error with ``category``, ``status``, and
        ``correlation_id`` set.
    """
    search_text: list[str] = _collect_strings(body) if body else []

    category: str | None = None
    correlation_id: str | None = None

    if body is not None:
        raw_category = body.get("category")
        if isinstance(raw_category, str):
            category = raw_category
        raw_cid = body.get("correlationId")
        if isinstance(raw_cid, str):
            correlation_id = raw_cid

    message: str
    error_category: str

    if status == 401:
        error_category = "config"
        message = (
            "HubSpot rejected HUBSPOT_ACCESS_TOKEN (401). Check the token "
            "value and that it belongs to the right HubSpot account."
        )
    elif status == 403:
        error_category = "config"
        message = (
            "HubSpot denied access (403). The token is missing a scope: it "
            "needs crm.objects.contacts.read and crm.objects.contacts.write."
        )
    elif status == 404:
        error_category = "config"
        message = (
            "HubSpot endpoint not found (404). Check _API_VERSION in "
            "services/hubspot_service.py."
        )
    elif status == 409:
        error_category = "conflict"
        message = (
            "HubSpot reports a conflicting contact for this email (409). "
            "Resolve or merge the duplicate in HubSpot, then retry."
        )
    elif status == 429:
        error_category = "transient"
        message = "HubSpot rate limit reached (429). Retry later."
    elif status >= 500:
        error_category = "transient"
        message = "HubSpot is temporarily unavailable. Retry later."
    elif "PROPERTY_DOESNT_EXIST" in " ".join(search_text):
        error_category = "config"
        message = (
            "A custom contact property is missing in HubSpot. Create "
            "lead_score, lead_status, property_type and "
            "preferred_location (internal names)."
        )
    elif "INVALID_EMAIL" in " ".join(search_text):
        error_category = "invalid"
        message = "HubSpot rejected the email address as invalid."
    elif "INVALID_OPTION" in " ".join(search_text):
        error_category = "config"
        message = (
            "A HubSpot dropdown property rejected the value. Use "
            "single-line text properties, or make the option internal "
            "values exactly hot, warm, cold."
        )
    else:
        error_category = "config"
        message = (
            f"HubSpot rejected the request "
            f"(HTTP {status}, category {category})."
        )

    if correlation_id is not None:
        message += f" [HubSpot correlation id: {correlation_id}]"

    return CRMError(
        message,
        category=error_category,
        status=status,
        correlation_id=correlation_id,
    )


def _parse_success(response: httpx.Response) -> tuple[str, bool | None]:
    """Parse a successful upsert response and extract the contact ID.

    Validates that the response body indicates a completed upsert with no
    errors.  If the response represents a failure (HTTP >= 400, non-empty
    ``errors`` list, or ``numErrors > 0``), delegates to
    ``_classify_failure``.

    Parameters
    ----------
    response : httpx.Response
        The HTTP response from the upsert call.

    Returns
    -------
    tuple[str, bool | None]
        A ``(contact_id, created)`` pair.  ``contact_id`` is the HubSpot
        contact ID as a string.  ``created`` is ``True`` if HubSpot created
        a new contact, ``False`` if it updated an existing one, or ``None``
        if the field was absent or not a boolean.

    Raises
    ------
    CRMError
        If the response indicates failure or has an unexpected structure.
    """
    body = _safe_json(response)

    # -- Explicit HTTP error -------------------------------------------------
    if response.status_code >= 400:
        raise _classify_failure(response.status_code, body)

    # -- 2xx / 207 with embedded errors --------------------------------------
    if body is not None:
        errors_list = body.get("errors")
        if isinstance(errors_list, list) and len(errors_list) > 0:
            # Try to extract a numeric status from the first error entry.
            first_error = errors_list[0]
            err_status: int = 400
            if isinstance(first_error, dict):
                raw_status = first_error.get("status")
                if isinstance(raw_status, int):
                    err_status = raw_status
            raise _classify_failure(err_status, body)

        num_errors = body.get("numErrors")
        if isinstance(num_errors, int) and num_errors > 0:
            # 2xx/207 with embedded errors and no usable error status: treat as 400.
            raise _classify_failure(400, body)

    # -- Successful response -------------------------------------------------
    if body is None:
        raise CRMError(
            "HubSpot returned an unexpected response.",
            category="config",
            status=response.status_code,
        )

    results = body.get("results")
    if not isinstance(results, list) or len(results) == 0:
        raise CRMError(
            "HubSpot returned an unexpected response.",
            category="config",
            status=response.status_code,
        )

    first_result = results[0]
    if not isinstance(first_result, dict):
        raise CRMError(
            "HubSpot returned an unexpected response.",
            category="config",
            status=response.status_code,
        )

    contact_id = first_result.get("id")
    if contact_id is None:
        raise CRMError(
            "HubSpot returned an unexpected response.",
            category="config",
            status=response.status_code,
        )

    created_raw = first_result.get("new")
    created: bool | None = created_raw if isinstance(created_raw, bool) else None

    return str(contact_id), created


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def upsert_contact(contact: HubSpotContact) -> dict[str, Any]:
    """Create or update a HubSpot contact via an idempotent batch upsert.

    The upsert is keyed on the contact's email address, so repeating the
    call with the same email is safe and the returned ``contact_id`` is
    stable across calls.

    In **dry-run mode** (the default when ``HUBSPOT_DRY_RUN`` is not set to
    ``"false"``), the contact is validated and the payload is prepared, but
    no HTTP request is made and no access token is required.

    In **live mode**, the contact is sent to HubSpot via a single batch
    upsert call.  If HubSpot returns a 409 (conflict), the request is
    retried once to settle create/create races.

    Callers must set ``hubspot_created`` in Supabase **only** when
    ``dry_run`` is ``False`` and ``contact_id`` is not ``None``.  Existing
    contacts get the standard and AI properties overwritten with the latest
    values, while omitted optional fields are left unchanged.

    Parameters
    ----------
    contact : HubSpotContact
        A validated HubSpot contact to upsert.

    Returns
    -------
    dict
        A result dictionary with keys:

        - ``message`` (str): Human-readable outcome description.
        - ``contact_id`` (str or None): The HubSpot contact ID, or ``None``
          in dry-run mode.
        - ``created`` (bool or None): ``True`` if a new contact was created,
          ``False`` if updated, ``None`` if unknown or dry-run.
        - ``status`` (str): One of ``"dry_run"``, ``"created"``,
          ``"updated"``, or ``"upserted"``.
        - ``dry_run`` (bool): ``True`` if no request was made.
        - ``properties_sent`` (dict): Summary of which property names were
          included (never values).

    Raises
    ------
    CRMError
        With an appropriate ``category`` if the upsert fails.
    """
    payload, summary = _build_payload(contact)

    # -- Dry-run mode: validate and prepare, but do not contact HubSpot ------
    if _is_dry_run():
        return {
            "message": (
                "DRY RUN: contact validated and prepared; nothing was sent "
                "to HubSpot. Set HUBSPOT_DRY_RUN=false to write to HubSpot."
            ),
            "contact_id": None,
            "created": None,
            "status": "dry_run",
            "dry_run": True,
            "properties_sent": summary,
        }

    # -- Live mode: send the upsert to HubSpot -------------------------------
    url = _UPSERT_PATH.format(version=_API_VERSION)

    try:
        with _get_http_client() as client:
            response = client.post(url, json=payload)

            # Retry once on 409 to settle create/create races.
            if response.status_code == 409:
                response = client.post(url, json=payload)
    except CRMError:
        raise
    except httpx.HTTPError as exc:
        raise CRMError(
            "Could not reach HubSpot (network error or timeout).",
            category="transient",
        ) from exc

    contact_id, created = _parse_success(response)

    if created is True:
        message = "HubSpot contact created"
        status_label = "created"
    elif created is False:
        message = "HubSpot contact updated"
        status_label = "updated"
    else:
        message = "HubSpot contact upserted"
        status_label = "upserted"

    return {
        "message": message,
        "contact_id": contact_id,
        "created": created,
        "status": status_label,
        "dry_run": False,
        "properties_sent": summary,
    }
