"""
routers/crm.py

HTTP routing layer for the HubSpot CRM integration.

This file contains zero business logic. Its sole responsibility is to:
    1. Accept HTTP requests and validate inputs at the routing layer.
    2. Delegate all HubSpot logic to services/hubspot_service.py.
    3. Map service return values and exceptions to appropriate HTTP responses.

Endpoint summary:
    POST /crm/contact  -- Upsert a contact in HubSpot by email address.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from models import HubSpotContact
from services.hubspot_service import CRMError, upsert_contact


router = APIRouter(prefix="/crm", tags=["CRM"])


# ===========================================================================
# RESPONSE MODELS
# ===========================================================================

class PropertiesSent(BaseModel):
    """Names of the HubSpot contact properties that were included in the upsert.

    Values are never included -- only the internal property names are listed,
    so this object can be logged or returned to callers without exposing PII.
    """

    standard: list[str]  # e.g. ["firstname", "lastname", "phone"]
    custom: list[str]    # e.g. ["lead_score", "lead_status", "property_type"]


class CRMContactResponse(BaseModel):
    """HTTP response body returned by POST /crm/contact.

    Reflects the outcome of the HubSpot batch upsert for a single contact.
    When ``dry_run`` is ``True``, no data was sent to HubSpot and
    ``contact_id`` is ``None``.  When ``dry_run`` is ``False`` and
    ``contact_id`` is not ``None``, the caller should update the
    ``hubspot_created`` flag via PATCH /leads/{id}/flags.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": (
                    "DRY RUN: contact validated and prepared; nothing was sent "
                    "to HubSpot. Set HUBSPOT_DRY_RUN=false to write to HubSpot."
                ),
                "contact_id": None,
                "created": None,
                "status": "dry_run",
                "dry_run": True,
                "properties_sent": {
                    "standard": ["firstname", "lastname", "phone"],
                    "custom": [
                        "lead_score",
                        "lead_status",
                        "property_type",
                        "preferred_location",
                    ],
                },
            }
        }
    )

    message: str
    contact_id: str | None
    created: bool | None
    status: str
    dry_run: bool
    properties_sent: PropertiesSent


# ===========================================================================
# ENDPOINT -- Upsert HubSpot Contact
# ===========================================================================

@router.post(
    "/contact",
    response_model=CRMContactResponse,
    summary="Upsert a contact in HubSpot CRM",
    responses={
        200: {
            "description": (
                "Contact successfully created, updated, or prepared in "
                "dry-run mode.  Check ``status`` and ``dry_run`` to "
                "distinguish outcomes."
            ),
        },
        409: {
            "description": (
                "A persistent conflicting contact exists in HubSpot.  "
                "Resolve or merge the duplicate in the HubSpot account, "
                "then retry.  Do not retry automatically."
            ),
        },
        422: {
            "description": (
                "The request body failed Pydantic validation, or HubSpot "
                "rejected the contact data (e.g. invalid email address).  "
                "Correct the data before retrying."
            ),
        },
        500: {
            "description": (
                "A configuration error or unexpected failure occurred "
                "(missing token, missing custom properties, bad API version, "
                "or an unclassified error).  Inspect the detail field and fix "
                "the configuration; retrying will not help."
            ),
        },
        502: {
            "description": (
                "HubSpot is temporarily unavailable or a network/timeout "
                "error occurred.  This response is retryable."
            ),
        },
    },
)
def upsert_contact_endpoint(contact: HubSpotContact) -> CRMContactResponse:
    """Create or update a HubSpot contact via an idempotent batch upsert.

    Accepts a fully-validated ``HubSpotContact`` and delegates to
    ``upsert_contact`` in ``services/hubspot_service.py``.  The endpoint is
    declared as a plain ``def`` (not ``async def``) because the underlying
    HTTP call to HubSpot is blocking; FastAPI runs synchronous handlers in a
    thread pool so the event loop is never blocked.

    **Properties**

    Three categories of contact data are handled differently:

    - *Identifier* -- ``email`` travels as the upsert key (``idProperty`` +
      ``id``) and is never included in the HubSpot ``properties`` dict.
    - *Standard properties* -- ``firstname``, ``lastname``, ``phone`` are
      built-in HubSpot fields present in every account.
    - *Custom properties* -- ``lead_score``, ``lead_status``,
      ``property_type``, and ``preferred_location`` must be created by hand
      in the HubSpot account (Settings > Properties) using the internal names
      shown here before this endpoint can write to them.

    Only non-empty, non-None values are sent; omitted optional fields are left
    unchanged on existing contacts.  ``properties_sent`` in the response lists
    every property name that was included, without exposing their values.

    **Idempotency**

    The upsert is keyed on the contact's email address.  Repeating the call
    with the same email is safe: HubSpot will create the contact on the first
    call and update it on subsequent calls.  The ``contact_id`` returned is
    stable across all calls for the same email.

    **Setting hubspot_created**

    After a successful live upsert, callers must update the lead record by
    calling ``PATCH /leads/{id}/flags`` to set ``hubspot_created = true``.
    This flag acts as an idempotency guard in the n8n pipeline.  Only set it
    when ``dry_run`` is ``false`` **and** ``contact_id`` is not ``null``; in
    dry-run mode nothing was sent to HubSpot and no flag should be set.

    **Status codes and retry guidance**

    - ``200`` -- success (or dry-run); check ``status`` and ``dry_run``.
    - ``502`` -- HubSpot or network is temporarily unavailable; retryable.
    - ``409`` -- conflicting duplicate contact in HubSpot; resolve manually,
      do not retry automatically.
    - ``422`` -- invalid input data; correct before retrying.
    - ``500`` -- configuration or unexpected error; retrying will not help.

    **Receives:**
        A ``HubSpotContact`` JSON request body:
        - ``email`` (required) -- contact's email address; upsert key.
        - ``firstname`` (required) -- contact's first name.
        - ``lastname`` (required) -- contact's last name.
        - ``phone`` (required) -- contact's phone number.
        - ``property_type`` (optional) -- desired property type.
        - ``preferred_location`` (optional) -- desired city or area.
        - ``lead_score`` (required) -- AI quality score in [0, 100].
        - ``lead_status`` (required) -- ``"hot"``, ``"warm"``, or ``"cold"``.

    **Returns:**
        A ``CRMContactResponse`` with:
        - ``message`` -- human-readable outcome description.
        - ``contact_id`` -- stable HubSpot contact ID, or ``null`` in dry-run.
        - ``created`` -- ``true`` if new, ``false`` if updated, ``null`` if
          unknown or dry-run.
        - ``status`` -- one of ``"created"``, ``"updated"``, ``"upserted"``,
          or ``"dry_run"``.
        - ``dry_run`` -- ``true`` when nothing was sent to HubSpot.
        - ``properties_sent`` -- names of properties included in the upsert.

    **Errors:**
        - ``422`` -- request body validation failed (FastAPI handles
          automatically), or HubSpot rejected the contact data.
        - ``409`` -- conflicting contact in HubSpot; manual resolution needed.
        - ``502`` -- HubSpot or network unavailable; retry after a delay.
        - ``500`` -- configuration error or unexpected failure.
    """
    try:
        return upsert_contact(contact)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CRMError as exc:
        status = {"invalid": 422, "conflict": 409, "transient": 502}.get(
            exc.category, 500
        )
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while syncing the contact.",
        ) from exc
