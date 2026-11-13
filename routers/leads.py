from uuid import UUID

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, model_validator

from models import LeadInput, LeadRecord
from services.pipeline_service import process_lead
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


# ---------------------------------------------------------------------------
# PIPELINE ENDPOINT MODELS
# ---------------------------------------------------------------------------

class ProcessLeadRequest(BaseModel):
    """Request body for POST /leads/process."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "lead": {
                    "name": "Sarah Mitchell",
                    "email": "sarah.mitchell@example.com",
                    "phone": "+1-415-555-0192",
                    "message": (
                        "My partner and I are looking for a three-bedroom home near "
                        "good schools and would like to move before summer."
                    ),
                    "property_type": "house",
                    "preferred_location": "Austin, TX",
                    "budget": "$450k-$550k",
                    "timeline": "within 3 months",
                },
            }
        }
    }

    id: UUID | None = None
    lead: LeadInput


class StepErrorModel(BaseModel):
    """Safe error details reported for a failed pipeline step."""

    category: str
    message: str


class StepReportModel(BaseModel):
    """API representation of one pipeline step outcome."""

    name: str
    status: str
    reason: str
    retryable: bool
    external_id: str | None
    flag_recorded: bool | None
    duration_ms: int
    error: StepErrorModel | None


class RetryAdviceModel(BaseModel):
    """Retry guidance derived from retryable failed pipeline steps."""

    recommended: bool
    steps: list[str]


class RepairActionModel(BaseModel):
    """A safe manual repair action for a sent-but-unflagged provider action."""

    action: str
    body: dict[str, bool]


class ProcessLeadResponse(BaseModel):
    """Complete, per-step report returned by POST /leads/process."""

    lead_id: str
    status: str
    persisted: bool
    created: bool | None
    idempotency_key_generated: bool
    category: str | None
    score: int | None
    steps: list[StepReportModel]
    retry: RetryAdviceModel
    repair: list[RepairActionModel]
    failed_step: str | None
    retryable: bool


@router.post(
    "/process",
    response_model=ProcessLeadResponse,
    responses={
        200: {
            "description": (
                "Lead persisted with a completed or partial pipeline report; "
                "inspect steps, retry, and repair."
            )
        },
        422: {"description": "Validation error (invalid request body or id)"},
        500: {"description": "Non-retryable pipeline failure"},
        502: {
            "description": (
                "Retryable score or storage failure; resend the same request."
            )
        },
    },
)
def process_lead_endpoint(
    request: ProcessLeadRequest,
    response: Response,
) -> ProcessLeadResponse:
    """Process, persist, and safely dispatch one lead through the pipeline.

    HTTP semantics:
        200 -- the lead was persisted and the pipeline is completed or partial;
        read ``steps``, ``retry``, and ``repair`` for individual outcomes.
        502 -- scoring or storage failed with a retryable error; resend the
        same request.
        500 -- a non-retryable pipeline failure occurred.
        422 -- the request body or id is invalid.

    The optional ``id`` is the idempotency key. Send the SAME id on every
    retry. If omitted, one is generated and returned; retrying without it
    creates a new lead. Dry-run steps are reported as skipped and never mark
    flags. Cold leads are persisted but receive no email, SMS, or CRM sync.
    A step whose reason is ``flag_update_failed`` needs the PATCH action in
    ``repair``, not a pipeline retry, because retrying would resend the
    external message.
    """
    try:
        result = process_lead(
            request.lead,
            lead_id=str(request.id) if request.id else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while processing the lead.",
        ) from exc

    if result.status == "failed":
        response.status_code = 502 if result.retry_steps else 500
    return ProcessLeadResponse(**result.to_dict())


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
