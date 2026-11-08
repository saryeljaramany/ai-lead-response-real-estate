"""
services/supabase_service.py

Supabase persistence layer for the real-estate lead system.

Stores and retrieves LeadRecord objects in the ``public.leads`` table via the
Supabase Data API (PostgREST) using a server-side secret key. Every database
failure is wrapped in a LeadStoreError whose *category* tells the HTTP router
and the n8n pipeline whether retrying is sensible.

Retry safety contract
    * Lead and score data are written once (first write wins via upsert-on-
      conflict logic).
    * The three delivery flags (email_sent, sms_sent, hubspot_created) can
      only transition from false to true; a false value is never written.
    * Callers must reuse the same lead id across retries.
"""

import base64
import json
import os
import uuid
from datetime import datetime, timezone

import httpx
from postgrest.exceptions import APIError
from pydantic import ValidationError
from supabase import Client, ClientOptions, create_client

from models import LeadInput, LeadRecord, LeadScore


# =========================================================================== #
# CONSTANTS                                                                    #
# =========================================================================== #

_TABLE = "leads"
_REQUEST_TIMEOUT_SECONDS = 10

_FLAG_COLUMNS = ("email_sent", "sms_sent", "hubspot_created")

_TRANSIENT_CODES = {
    "PGRST000", "PGRST001", "PGRST002", "PGRST003",
    "57014", "53300", "40001", "40P01",
}

_LEAD_INPUT_FIELDS = (
    "name", "email", "phone", "message",
    "property_type", "preferred_location", "budget", "timeline",
)

_LEAD_SCORE_FIELDS = (
    "score", "category", "reasoning", "suggested_action", "priority_level",
)


# =========================================================================== #
# EXCEPTION                                                                    #
# =========================================================================== #

class LeadStoreError(RuntimeError):
    """Classified persistence error for the leads table.

    Parameters
    ----------
    message:
        A safe, human-readable description of the failure. Never contains
        credentials, URLs, row data, or internal Supabase error details.
    category:
        One of ``"config"``, ``"transient"``, or ``"not_found"``.

        * ``"config"`` -- setup, credentials, schema or permission problem;
          retrying will not help.
        * ``"transient"`` -- Supabase or the network failed; a retry is
          reasonable.
        * ``"not_found"`` -- the requested lead id does not exist.
    db_code:
        The raw Supabase / PostgreSQL error code when available (e.g.
        ``"23505"`` for unique-violation).
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        db_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category: str = category
        self.db_code: str | None = db_code


# =========================================================================== #
# PRIVATE HELPERS                                                              #
# =========================================================================== #

def _require_env(name: str) -> str:
    """Read an environment variable at call time and return its stripped value.

    Parameters
    ----------
    name:
        The environment variable name (e.g. ``"SUPABASE_URL"``).

    Returns
    -------
    str
        The non-empty, stripped value.

    Raises
    ------
    LeadStoreError
        ``category="config"`` when the variable is unset or blank.
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise LeadStoreError(
            f"{name} is missing or empty. Set it in .env and restart the server.",
            category="config",
        )
    return value.strip()


def _check_key_kind(key: str) -> None:
    """Reject publishable and anon keys that cannot write to leads.

    Parameters
    ----------
    key:
        The raw ``SUPABASE_KEY`` value.

    Raises
    ------
    LeadStoreError
        ``category="config"`` when the key is a publishable or anon key.
    """
    if key.startswith("sb_publishable_"):
        raise LeadStoreError(
            "SUPABASE_KEY is a publishable key. Use a secret key "
            "(sb_secret_...) or the legacy service_role key from "
            "Supabase > Settings > API Keys.",
            category="config",
        )

    segments = key.split(".")
    if len(segments) == 3:
        payload_segment = segments[1]
        # Restore base64url padding.
        padded = payload_segment + "=" * (-len(payload_segment) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded)
            claims = json.loads(decoded)
            role = claims.get("role")
        except Exception:
            # Unparseable JWT payload -- let Supabase judge it.
            return
        if role == "anon":
            raise LeadStoreError(
                "SUPABASE_KEY is an anon key. Use a secret key "
                "(sb_secret_...) or the legacy service_role key from "
                "Supabase > Settings > API Keys.",
                category="config",
            )


def _get_client() -> Client:
    """Create and return a fresh Supabase client.

    Reads ``SUPABASE_URL`` and ``SUPABASE_KEY`` from the environment on
    every call. The client is never cached so that hot-reloaded credentials
    take effect immediately.

    Returns
    -------
    Client
        A configured ``supabase.Client`` ready for Data API calls.

    Raises
    ------
    LeadStoreError
        ``category="config"`` when the URL or key is invalid or when the
        SDK fails to initialise.
    """
    url = _require_env("SUPABASE_URL").rstrip("/")

    if not url.startswith("https://"):
        raise LeadStoreError(
            "SUPABASE_URL must be the project root, "
            "e.g. https://<project-ref>.supabase.co",
            category="config",
        )
    if "/rest/v1" in url:
        raise LeadStoreError(
            "SUPABASE_URL must be the project root, "
            "e.g. https://<project-ref>.supabase.co",
            category="config",
        )

    key = _require_env("SUPABASE_KEY")
    _check_key_kind(key)

    try:
        return create_client(
            url,
            key,
            options=ClientOptions(
                postgrest_client_timeout=_REQUEST_TIMEOUT_SECONDS,
                auto_refresh_token=False,
                persist_session=False,
            ),
        )
    except Exception as exc:
        raise LeadStoreError(
            f"Could not initialise the Supabase client: "
            f"{type(exc).__name__}.",
            category="config",
        ) from exc


def _to_uuid(value: str) -> str:
    """Validate and normalise a string to a canonical lowercase UUID.

    Parameters
    ----------
    value:
        The candidate UUID string.

    Returns
    -------
    str
        The canonical lowercase UUID representation.

    Raises
    ------
    ValueError
        When *value* is not a valid UUID.
    """
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        raise ValueError("lead id must be a valid UUID.")


def _record_to_row(record: LeadRecord) -> dict:
    """Flatten a LeadRecord into the 18-column dict expected by public.leads.

    Parameters
    ----------
    record:
        The in-memory lead record to persist.

    Returns
    -------
    dict
        A dictionary with exactly the 18 database column keys.

    Raises
    ------
    ValueError
        Propagated from ``_to_uuid`` when the record id is invalid.
    """
    created_at = record.created_at
    if created_at is not None:
        # Treat naive datetimes as UTC.
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        created_at_str: str | None = created_at.isoformat()
    else:
        created_at_str = None

    return {
        "id": _to_uuid(record.id),
        "created_at": created_at_str,
        "name": record.lead.name,
        "email": record.lead.email,
        "phone": record.lead.phone,
        "message": record.lead.message,
        "property_type": record.lead.property_type,
        "preferred_location": record.lead.preferred_location,
        "budget": record.lead.budget,
        "timeline": record.lead.timeline,
        "score": record.score.score,
        "category": record.score.category,
        "reasoning": record.score.reasoning,
        "suggested_action": record.score.suggested_action,
        "priority_level": record.score.priority_level,
        "email_sent": record.email_sent,
        "sms_sent": record.sms_sent,
        "hubspot_created": record.hubspot_created,
    }


def _row_to_record(row: dict) -> LeadRecord:
    """Reconstruct a LeadRecord from a database row dictionary.

    Parameters
    ----------
    row:
        A dictionary returned by the Supabase Data API containing all 18
        expected columns.

    Returns
    -------
    LeadRecord
        The rehydrated domain object. ``LeadScore.category`` is re-derived
        from ``score`` by the model validator, so the returned object always
        uses the current threshold constants.

    Raises
    ------
    LeadStoreError
        ``category="config"`` when a required column is missing or the data
        fails Pydantic validation (schema drift).
    """
    all_columns = (
        "id", "created_at",
        *_LEAD_INPUT_FIELDS,
        *_LEAD_SCORE_FIELDS,
        *_FLAG_COLUMNS,
    )
    for col in all_columns:
        if col not in row:
            raise LeadStoreError(
                f"The leads row is missing column '{col}'; the database "
                f"schema does not match models/lead.py.",
                category="config",
            )

    try:
        return LeadRecord(
            id=row["id"],
            lead=LeadInput(
                name=row["name"],
                email=row["email"],
                phone=row["phone"],
                message=row["message"],
                property_type=row["property_type"],
                preferred_location=row["preferred_location"],
                budget=row["budget"],
                timeline=row["timeline"],
            ),
            score=LeadScore(
                score=row["score"],
                category=row["category"],
                reasoning=row["reasoning"],
                suggested_action=row["suggested_action"],
                priority_level=row["priority_level"],
            ),
            email_sent=row["email_sent"],
            sms_sent=row["sms_sent"],
            hubspot_created=row["hubspot_created"],
            created_at=row["created_at"],
        )
    except ValidationError as exc:
        field_locations = ", ".join(
            ".".join(str(part) for part in e["loc"]) for e in exc.errors()
        )
        raise LeadStoreError(
            f"Stored lead failed model validation for fields: "
            f"{field_locations}.",
            category="config",
        ) from exc


def _classify_error(exc: Exception) -> LeadStoreError:
    """Wrap an arbitrary exception in a categorised LeadStoreError.

    Parameters
    ----------
    exc:
        The exception raised during a Supabase operation.

    Returns
    -------
    LeadStoreError
        A classified error with a safe message and an appropriate category.
    """
    if isinstance(exc, LeadStoreError):
        return exc

    if isinstance(exc, APIError):
        code = str(exc.code) if exc.code is not None else None
        suffix = f" (Supabase error {code})" if code else ""

        if code == "42501":
            return LeadStoreError(
                "Supabase denied access to public.leads. Check that "
                "db/migrations/001_create_leads.sql ran (it grants "
                "service_role) and that SUPABASE_KEY is a secret or "
                "service_role key, not a publishable or anon key."
                + suffix,
                category="config",
                db_code=code,
            )

        if code in ("42P01", "PGRST205"):
            return LeadStoreError(
                "Table public.leads was not found. Run "
                "db/migrations/001_create_leads.sql in the Supabase "
                "SQL Editor." + suffix,
                category="config",
                db_code=code,
            )

        if code == "PGRST301":
            return LeadStoreError(
                "Supabase rejected SUPABASE_KEY. Check the key value."
                + suffix,
                category="config",
                db_code=code,
            )

        if code is not None and code in _TRANSIENT_CODES:
            return LeadStoreError(
                "Supabase is temporarily unavailable. Retry later."
                + suffix,
                category="transient",
                db_code=code,
            )

        if code in ("23502", "23514", "22P02", "22001"):
            return LeadStoreError(
                "The database rejected the data. The schema and "
                "models/lead.py are out of sync." + suffix,
                category="config",
                db_code=code,
            )

        if code is not None and code.isdigit() and len(code) == 3 and 500 <= int(code) <= 599:
            return LeadStoreError(
                "Supabase is temporarily unavailable. Retry later."
                + suffix,
                category="transient",
                db_code=code,
            )

        # Fallback: include only exc.message (never details).
        msg = str(exc.message) if exc.message is not None else "unknown error"
        return LeadStoreError(
            f"Supabase rejected the request: {msg}" + suffix,
            category="config",
            db_code=code,
        )

    if isinstance(exc, httpx.HTTPError):
        return LeadStoreError(
            "Could not reach Supabase (network error or timeout).",
            category="transient",
        )

    return LeadStoreError(
        f"Unexpected error while calling Supabase: {type(exc).__name__}.",
        category="config",
    )


def _execute(builder):
    """Execute a PostgREST query builder and return the result.

    Parameters
    ----------
    builder:
        A Supabase query builder (e.g. from ``client.table(...).select(...)``).

    Returns
    -------
    object
        The API response object whose ``.data`` attribute contains the rows.

    Raises
    ------
    LeadStoreError
        A classified error wrapping whatever the SDK raised.
    """
    try:
        return builder.execute()
    except Exception as exc:
        raise _classify_error(exc) from exc


# =========================================================================== #
# PUBLIC FUNCTIONS                                                             #
# =========================================================================== #

def store_lead(record: LeadRecord) -> tuple[LeadRecord, bool]:
    """Persist a LeadRecord, creating it if new or merging flags if it exists.

    First-write-wins semantics: lead and score data are written only on the
    initial insert. If the id already exists (unique-violation ``23505``),
    only the delivery flags that are ``True`` on *record* are OR-merged into
    the existing row. A ``False`` flag is never written, so flags can only
    move from false to true. Callers must reuse the same id on retries.

    Parameters
    ----------
    record:
        The lead record to store. Its ``id`` must be a valid UUID.

    Returns
    -------
    tuple[LeadRecord, bool]
        ``(stored_record, created)`` where *created* is ``True`` when a new
        row was inserted and ``False`` when an existing row was updated.

    Raises
    ------
    LeadStoreError
        Classified as ``"config"`` or ``"transient"`` depending on the root
        cause.
    ValueError
        When ``record.id`` is not a valid UUID.
    """
    row = _record_to_row(record)
    lead_id = row["id"]
    client = _get_client()

    # ------------------------------------------------------------------
    # Step 1 -- attempt insert
    # ------------------------------------------------------------------
    created = False
    insert_data = None
    try:
        result = client.table(_TABLE).insert(row).execute()
        created = True
        if result.data:
            insert_data = result.data[0]
    except APIError as exc:
        if str(exc.code) == "23505":
            created = False
        else:
            raise _classify_error(exc) from exc
    except Exception as exc:
        raise _classify_error(exc) from exc

    # ------------------------------------------------------------------
    # Step 2 -- build the returned record
    # ------------------------------------------------------------------
    if created and insert_data is not None:
        return _row_to_record(insert_data), True

    if not created:
        # OR-merge flags that are True on the incoming record.
        flag_values = {
            flag: True for flag in _FLAG_COLUMNS if row[flag] is True
        }
        if flag_values:
            _execute(
                client.table(_TABLE).update(flag_values).eq("id", lead_id)
            )

    # Read the authoritative row back.
    select_result = _execute(
        client.table(_TABLE).select("*").eq("id", lead_id).limit(1)
    )
    if not select_result.data:
        raise LeadStoreError(
            "Insert conflicted on a constraint other than the lead id; "
            "check the database schema.",
            category="config",
        )

    return _row_to_record(select_result.data[0]), created


def get_lead(lead_id: str) -> LeadRecord:
    """Fetch a single lead by its UUID.

    Parameters
    ----------
    lead_id:
        The UUID of the lead to retrieve.

    Returns
    -------
    LeadRecord
        The rehydrated domain object.

    Raises
    ------
    LeadStoreError
        ``category="not_found"`` when no row matches *lead_id*.
        ``category="config"`` or ``"transient"`` for other failures.
    ValueError
        When *lead_id* is not a valid UUID.
    """
    canonical_id = _to_uuid(lead_id)
    client = _get_client()
    result = _execute(
        client.table(_TABLE).select("*").eq("id", canonical_id).limit(1)
    )
    if not result.data:
        raise LeadStoreError(
            f"Lead {canonical_id} was not found.",
            category="not_found",
        )
    return _row_to_record(result.data[0])


def update_flags(
    lead_id: str,
    *,
    email_sent: bool = False,
    sms_sent: bool = False,
    hubspot_created: bool = False,
) -> LeadRecord:
    """Set one or more delivery flags to True on an existing lead.

    Flags can only move from false to true. Passing ``False`` (the default)
    for a flag leaves it unchanged; it is never written to the database.

    Parameters
    ----------
    lead_id:
        The UUID of the lead to update.
    email_sent:
        Set to ``True`` to mark the follow-up email as sent.
    sms_sent:
        Set to ``True`` to mark the follow-up SMS as sent.
    hubspot_created:
        Set to ``True`` to mark the HubSpot contact as created.

    Returns
    -------
    LeadRecord
        The updated lead record read back from the database.

    Raises
    ------
    ValueError
        When no flag is set to ``True``.
    LeadStoreError
        ``category="not_found"`` when no row matches *lead_id*.
        ``category="config"`` or ``"transient"`` for other failures.
    """
    values: dict[str, bool] = {}
    if email_sent:
        values["email_sent"] = True
    if sms_sent:
        values["sms_sent"] = True
    if hubspot_created:
        values["hubspot_created"] = True

    if not values:
        raise ValueError(
            "At least one flag must be true. Flags can only be set to true."
        )

    canonical_id = _to_uuid(lead_id)
    client = _get_client()

    result = _execute(
        client.table(_TABLE).update(values).eq("id", canonical_id)
    )

    if not result.data:
        raise LeadStoreError(
            f"Lead {canonical_id} was not found.",
            category="not_found",
        )

    return _row_to_record(result.data[0])
