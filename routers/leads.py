from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from models import LeadRecord
from services.supabase_service import LeadStoreError, get_lead, store_lead, update_flags

# Note: This file will be expanded with CRUD operations and lead-specific logic in later phases.
router = APIRouter(prefix="/leads", tags=["Leads"])


@router.get("/test")
def test_router():
    """
    Test endpoint to confirm the leads router is properly connected.
    """
    return {"message": "Leads router is working correctly"}


# ---------------------------------------------------------------------------
# HTTP-LAYER MODELS
# ---------------------------------------------------------------------------

class StoreLeadResponse(BaseModel):
    """Response envelope returned by POST /leads/store."""

    created: bool
    record: LeadRecord


class LeadFlagsUpdate(BaseModel):
    """
    Request body for PATCH /leads/{lead_id}/flags.

    Every provided flag must be True; flags can never be reset to False
    through the API. At least one flag must be provided.
    """

    email_sent: bool | None = None
    sms_sent: bool | None = None
    hubspot_created: bool | None = None

    @model_validator(mode="after")
    def at_least_one_true_flag(self) -> "LeadFlagsUpdate":
        """Require at least one flag and reject any flag that is not True."""
        provided = [
            v for v in (self.email_sent, self.sms_sent, self.hubspot_created)
            if v is not None
        ]
        if not provided or any(v is not True for v in provided):
            raise ValueError(
                "Provide at least one flag, and only set flags to true; "
                "flags are never reset through the API."
            )
        return self


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _to_http(exc: LeadStoreError) -> HTTPException:
    """Map a LeadStoreError category to the appropriate HTTPException.

    Parameters
    ----------
    exc:
        The classified storage error.

    Returns
    -------
    HTTPException
        404 for not_found, 502 for transient, 500 for everything else.
    """
    status_code = {"not_found": 404, "transient": 502}.get(exc.category, 500)
    return HTTPException(status_code=status_code, detail=str(exc))


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

_COMMON_RESPONSES = {
    422: {"description": "Validation error (invalid request body or UUID)"},
    500: {"description": "Unexpected server-side or configuration error"},
    502: {"description": "Supabase is temporarily unavailable; safe to retry"},
}


@router.post(
    "/store",
    response_model=StoreLeadResponse,
    responses={
        200: {"description": "Lead stored successfully"},
        422: _COMMON_RESPONSES[422],
        500: _COMMON_RESPONSES[500],
        502: _COMMON_RESPONSES[502],
    },
)
def store_lead_endpoint(record: LeadRecord) -> StoreLeadResponse:
    """Store a fully-scored LeadRecord in the database.

    Receives:
        A complete LeadRecord in the request body, including a pre-assigned
        UUID ``id``, the nested ``lead`` (LeadInput) and ``score``
        (LeadScore) objects, and the three optional delivery flags.

    Returns:
        StoreLeadResponse with ``created=True`` and the persisted record when
        a new row was inserted, or ``created=False`` and the existing record
        when the ``id`` was already present.

    Idempotency:
        The ``id`` field makes this call idempotent. Callers MUST send the
        same ``id`` on every retry attempt. When a duplicate ``id`` is
        received, the stored lead and score data are never overwritten; only
        delivery flags that are ``True`` in the request are merged into the
        existing row (OR semantics). A duplicate-id response returns the
        authoritative stored record with ``created=false``.

    Errors:
        422 -- the request body or UUID failed validation.
        500 -- a configuration or unexpected server error occurred.
        502 -- Supabase is temporarily unavailable; retrying later is safe.
    """
    try:
        stored, created = store_lead(record)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LeadStoreError as exc:
        raise _to_http(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while storing the lead.",
        ) from exc
    return StoreLeadResponse(created=created, record=stored)


@router.get(
    "/{lead_id}",
    response_model=LeadRecord,
    responses={
        200: {"description": "Lead record returned successfully"},
        404: {"description": "No lead found for the given id"},
        422: _COMMON_RESPONSES[422],
        500: _COMMON_RESPONSES[500],
        502: _COMMON_RESPONSES[502],
    },
)
def get_lead_endpoint(lead_id: UUID) -> LeadRecord:
    """Retrieve a single lead by its UUID.

    Receives:
        ``lead_id`` -- a valid UUID in the URL path.

    Returns:
        The full LeadRecord for the requested id.

    Errors:
        404 -- no lead exists with the given id.
        422 -- the path segment is not a valid UUID.
        500 -- a configuration or unexpected server error occurred.
        502 -- Supabase is temporarily unavailable; retrying later is safe.
    """
    try:
        record = get_lead(str(lead_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LeadStoreError as exc:
        raise _to_http(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while retrieving the lead.",
        ) from exc
    return record


@router.patch(
    "/{lead_id}/flags",
    response_model=LeadRecord,
    responses={
        200: {"description": "Flags updated; updated lead record returned"},
        404: {"description": "No lead found for the given id"},
        422: _COMMON_RESPONSES[422],
        500: _COMMON_RESPONSES[500],
        502: _COMMON_RESPONSES[502],
    },
)
def update_flags_endpoint(lead_id: UUID, body: LeadFlagsUpdate) -> LeadRecord:
    """Set one or more delivery flags to True on an existing lead.

    Receives:
        ``lead_id`` -- a valid UUID in the URL path.
        Request body -- a LeadFlagsUpdate object; at least one flag must be
        present and every provided flag must be True (flags are never reset).

    Returns:
        The updated LeadRecord read back from the database after the patch.

    Errors:
        404 -- no lead exists with the given id.
        422 -- the path segment is not a valid UUID, or the flag payload
               is missing, empty, or contains a False value.
        500 -- a configuration or unexpected server error occurred.
        502 -- Supabase is temporarily unavailable; retrying later is safe.
    """
    flag_kwargs = {
        flag: True
        for flag, value in {
            "email_sent": body.email_sent,
            "sms_sent": body.sms_sent,
            "hubspot_created": body.hubspot_created,
        }.items()
        if value is True
    }
    try:
        record = update_flags(str(lead_id), **flag_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LeadStoreError as exc:
        raise _to_http(exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while updating lead flags.",
        ) from exc
    return record
