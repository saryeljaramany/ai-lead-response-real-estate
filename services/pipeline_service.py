"""Synchronous, retry-safe orchestration for one real-estate lead.

The pipeline deliberately keeps persistence authoritative.  Once a record has
been stored, all outbound work is based on the stored lead, score, and flags.
Dependencies are injectable so callers can exercise every outcome without
network, database, or provider access.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import dataclasses
import logging
import time
import typing
import uuid

import openai

from models import LeadInput, LeadRecord
from services.gmail_service import EmailSendError, send_email
from services.hubspot_service import CRMError, contact_from_record, upsert_contact
from services.llm_service import generate_email, generate_sms, score_lead
from services.phone_service import normalize_phone
from services.supabase_service import LeadStoreError, store_lead, update_flags
from services.twilio_service import SMSSendError, send_sms


# ---------------------------------------------------------------------------
# Constants and logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_MAX_FLAG_ATTEMPTS = 2
_OUTBOUND_DETAILS: dict[str, tuple[str, str, str]] = {
    "email": ("email_sent", "message_id", "sent"),
    "sms": ("sms_sent", "message_id", "sent"),
    "crm": ("hubspot_created", "contact_id", "synced"),
}


# ---------------------------------------------------------------------------
# Public report contracts
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StepError:
    """A safe, categorised explanation of one pipeline step failure."""

    category: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return this error as a JSON-serialisable dictionary."""
        return {"category": self.category, "message": self.message}


@dataclasses.dataclass(frozen=True)
class StepResult:
    """Outcome details for one of the six fixed pipeline steps."""

    name: str
    status: typing.Literal["succeeded", "skipped", "failed"]
    reason: str
    retryable: bool = False
    external_id: str | None = None
    flag_recorded: bool | None = None
    duration_ms: int = 0
    error: StepError | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        """Return this result as plain JSON-serialisable data."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "retryable": self.retryable,
            "external_id": self.external_id,
            "flag_recorded": self.flag_recorded,
            "duration_ms": self.duration_ms,
            "error": self.error.to_dict() if self.error is not None else None,
        }


@dataclasses.dataclass
class PipelineResult:
    """The full report for a single lead and its six pipeline steps."""

    lead_id: str
    status: typing.Literal["completed", "partial", "failed"]
    persisted: bool
    created: bool | None
    idempotency_key_generated: bool
    category: str | None
    score: int | None
    steps: list[StepResult]
    retry_steps: list[str]
    repair: list[dict[str, typing.Any]]
    failed_step: str | None

    def to_dict(self) -> dict[str, typing.Any]:
        """Return the report in the stable API response shape."""
        return {
            "lead_id": self.lead_id,
            "status": self.status,
            "persisted": self.persisted,
            "created": self.created,
            "idempotency_key_generated": self.idempotency_key_generated,
            "category": self.category,
            "score": self.score,
            "steps": [step.to_dict() for step in self.steps],
            "retry": {
                "recommended": bool(self.retry_steps),
                "steps": list(self.retry_steps),
            },
            "repair": [
                {"action": item["action"], "body": dict(item["body"])}
                for item in self.repair
            ],
            "failed_step": self.failed_step,
            "retryable": bool(self.retry_steps),
        }


@dataclasses.dataclass(frozen=True)
class PipelineDeps:
    """All collaborators required by :func:`run_pipeline`.

    There are intentionally no defaults: tests and alternate runtimes can pass
    fully controlled collaborators, while production uses :func:`_default_deps`.
    """

    normalize_phone: typing.Callable[[object], typing.Any]
    score_lead: typing.Callable[[LeadInput], typing.Any]
    store_lead: typing.Callable[[LeadRecord], tuple[LeadRecord, bool]]
    update_flags: typing.Callable[..., LeadRecord]
    generate_email: typing.Callable[[LeadInput, typing.Any], typing.Any]
    send_email: typing.Callable[[typing.Any], object]
    generate_sms: typing.Callable[[LeadInput, typing.Any], typing.Any]
    send_sms: typing.Callable[[typing.Any], object]
    contact_from_record: typing.Callable[[LeadRecord], typing.Any]
    upsert_contact: typing.Callable[[typing.Any], object]


# ---------------------------------------------------------------------------
# Dependency and report helpers
# ---------------------------------------------------------------------------

def _default_deps() -> PipelineDeps:
    """Build production dependencies from the current module globals.

    Looking up globals when this function is called keeps monkeypatching and
    dependency replacement straightforward without triggering service work at
    import time.
    """
    return PipelineDeps(
        normalize_phone=normalize_phone,
        score_lead=score_lead,
        store_lead=store_lead,
        update_flags=update_flags,
        generate_email=generate_email,
        send_email=send_email,
        generate_sms=generate_sms,
        send_sms=send_sms,
        contact_from_record=contact_from_record,
        upsert_contact=upsert_contact,
    )


def _duration_ms(started_at: float) -> int:
    """Return a non-negative whole-millisecond duration from ``perf_counter``."""
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _skipped_step(name: str, reason: str, started_at: float) -> StepResult:
    """Create a measured, non-retryable skipped result."""
    return StepResult(
        name=name,
        status="skipped",
        reason=reason,
        duration_ms=_duration_ms(started_at),
    )


def _log_failure(lead_id: str, step: str, error: StepError, error_name: str) -> None:
    """Emit the one privacy-safe warning required for a failed step."""
    logger.warning(
        "pipeline step failed: lead_id=%s step=%s category=%s error=%s",
        lead_id,
        step,
        error.category,
        error_name,
    )


def _failed_step(
    lead_id: str,
    name: str,
    reason: str,
    error: StepError,
    retryable: bool,
    started_at: float,
    *,
    error_name: str,
    external_id: str | None = None,
    flag_recorded: bool | None = None,
) -> StepResult:
    """Log and create one failed step result without exposing customer data."""
    _log_failure(lead_id, name, error, error_name)
    return StepResult(
        name=name,
        status="failed",
        reason=reason,
        retryable=retryable,
        external_id=external_id,
        flag_recorded=flag_recorded,
        duration_ms=_duration_ms(started_at),
        error=error,
    )


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _classify(exc: Exception, step: str, stage: str) -> tuple[StepError, bool]:
    """Convert a collaborator exception into a safe step error and retry hint."""
    if isinstance(exc, (LeadStoreError, SMSSendError, EmailSendError, CRMError)):
        category = exc.category
        return StepError(category=category, message=str(exc)), category == "transient"

    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return (
            StepError(
                category="transient",
                message=(
                    "The AI provider is temporarily unavailable or rate-limited. "
                    "Retry later."
                ),
            ),
            True,
        )

    if isinstance(exc, openai.APIStatusError):
        return (
            StepError(
                category="config",
                message=(
                    f"The AI provider rejected the request (HTTP {exc.status_code}). "
                    "Check OPENROUTER_API_KEY, OPENROUTER_MODEL and account credits."
                ),
            ),
            False,
        )

    if isinstance(exc, openai.APIError):
        return (
            StepError(
                category="transient",
                message="The AI provider returned an error. Retry later.",
            ),
            True,
        )

    if isinstance(exc, ValueError):
        if stage in {"score", "generate"}:
            return (
                StepError(
                    category="model_output",
                    message="The AI model returned unusable output. A retry may succeed.",
                ),
                True,
            )
        return (
            StepError(
                category="invalid_data",
                message="The data could not be prepared for this step.",
            ),
            False,
        )

    if isinstance(exc, RuntimeError):
        if stage in {"score", "generate"}:
            return (
                StepError(
                    category="config",
                    message=(
                        "LLM configuration problem: check OPENROUTER_API_KEY "
                        "and OPENROUTER_MODEL."
                    ),
                ),
                False,
            )
        return (
            StepError(
                category="config",
                message=f"Configuration problem in step '{step}'.",
            ),
            False,
        )

    return (
        StepError(
            category="unexpected",
            message=f"Unexpected error in step '{step}' ({type(exc).__name__}).",
        ),
        False,
    )


# ---------------------------------------------------------------------------
# Outbound orchestration
# ---------------------------------------------------------------------------

def _run_outbound_step(
    name: str,
    stored: LeadRecord,
    cold: bool,
    deps: PipelineDeps,
    *,
    precheck: typing.Callable[[], str | None],
    prepare: typing.Callable[[], typing.Any],
    send: typing.Callable[[typing.Any], object],
    prepare_stage: str,
) -> tuple[StepResult, LeadRecord, dict[str, typing.Any] | None]:
    """Run one independently recoverable outbound action.

    The returned record is the original stored record unless a successful flag
    update returns a newer authoritative record.  A repair item is returned
    only when a live provider action could not be recorded in storage.
    """
    started_at = time.perf_counter()
    flag_name, external_id_key, success_reason = _OUTBOUND_DETAILS[name]
    lead_id = stored.id

    if cold:
        return _skipped_step(name, "cold_lead", started_at), stored, None

    if getattr(stored, flag_name) is True:
        return _skipped_step(name, "already_done", started_at), stored, None

    try:
        skip_reason = precheck()
    except Exception as exc:  # Collaborators are deliberately isolated.
        error, retryable = _classify(exc, name, "precheck")
        return (
            _failed_step(
                lead_id,
                name,
                error.category,
                error,
                retryable,
                started_at,
                error_name=type(exc).__name__,
            ),
            stored,
            None,
        )

    if skip_reason is not None:
        return _skipped_step(name, skip_reason, started_at), stored, None

    try:
        payload = prepare()
    except Exception as exc:  # Preparation failures must not block other steps.
        error, retryable = _classify(exc, name, prepare_stage)
        return (
            _failed_step(
                lead_id,
                name,
                error.category,
                error,
                retryable,
                started_at,
                error_name=type(exc).__name__,
            ),
            stored,
            None,
        )

    try:
        response = send(payload)
    except Exception as exc:  # Send failures must not block other steps.
        error, retryable = _classify(exc, name, "send")
        return (
            _failed_step(
                lead_id,
                name,
                error.category,
                error,
                retryable,
                started_at,
                error_name=type(exc).__name__,
            ),
            stored,
            None,
        )

    if not isinstance(response, dict):
        error = StepError(
            category="unexpected",
            message=f"Unexpected response from step '{name}'.",
        )
        return (
            _failed_step(
                lead_id,
                name,
                "unexpected",
                error,
                False,
                started_at,
                error_name="UnexpectedResponse",
            ),
            stored,
            None,
        )

    if response.get("dry_run") is not False:
        return _skipped_step(name, "dry_run", started_at), stored, None

    external_id = response.get(external_id_key)
    if not isinstance(external_id, str) or not external_id.strip():
        error = StepError(
            category="unexpected",
            message=f"Step '{name}' returned no usable external id.",
        )
        return (
            _failed_step(
                lead_id,
                name,
                "missing_external_id",
                error,
                False,
                started_at,
                error_name="MissingExternalId",
            ),
            stored,
            None,
        )

    last_error: StepError | None = None
    last_error_name = "FlagUpdateError"
    for _attempt in range(_MAX_FLAG_ATTEMPTS):
        try:
            updated_record = deps.update_flags(lead_id, **{flag_name: True})
        except Exception as exc:  # A provider success still needs repair on failure.
            last_error, _ignored_retryable = _classify(exc, name, "flag")
            last_error_name = type(exc).__name__
            continue

        return (
            StepResult(
                name=name,
                status="succeeded",
                reason=success_reason,
                external_id=external_id,
                flag_recorded=True,
                duration_ms=_duration_ms(started_at),
            ),
            updated_record,
            None,
        )

    if last_error is None:
        last_error = StepError(
            category="unexpected",
            message=f"Unexpected error in step '{name}' (FlagUpdateError).",
        )

    result = _failed_step(
        lead_id,
        name,
        "flag_update_failed",
        last_error,
        False,
        started_at,
        error_name=last_error_name,
        external_id=external_id,
        flag_recorded=False,
    )
    logger.error(
        "sent but not flagged: lead_id=%s step=%s category=%s error=%s",
        lead_id,
        name,
        last_error.category,
        last_error_name,
    )
    return (
        result,
        stored,
        {
            "action": f"PATCH /leads/{lead_id}/flags",
            "body": {flag_name: True},
        },
    )


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    lead: LeadInput,
    *,
    lead_id: str | None = None,
    deps: PipelineDeps | None = None,
) -> PipelineResult:
    """Run one lead through intake, scoring, storage, and outbound delivery.

    The function is synchronous by design for use from a normal FastAPI
    endpoint, which FastAPI executes in its thread pool.  Aside from an
    invalid supplied lead id, it returns a structured report for every
    collaborator outcome and always includes all six fixed step entries.
    """
    if lead_id is None:
        canonical_lead_id = str(uuid.uuid4())
        idempotency_key_generated = True
    else:
        try:
            canonical_lead_id = str(uuid.UUID(lead_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("lead id must be a valid UUID.") from exc
        idempotency_key_generated = False

    active_deps = deps if deps is not None else _default_deps()
    steps: list[StepResult] = []
    repair: list[dict[str, typing.Any]] = []

    # Intake is deliberately non-failing: retain the validated raw phone when
    # normalisation is unavailable or unsuitable, and let the SMS precheck
    # evaluate it again on a later retry.
    intake_started_at = time.perf_counter()
    intake_lead = lead
    intake_reason = "invalid_phone"
    try:
        phone = active_deps.normalize_phone(lead.phone)
        normalised_phone = phone.e164
        phone_reason = phone.reason
        if normalised_phone:
            if normalised_phone != lead.phone:
                intake_lead = lead.model_copy(update={"phone": normalised_phone})
                intake_reason = "normalized"
            else:
                intake_reason = "unchanged"
        elif phone_reason == "needs_country_code":
            intake_reason = "needs_country_code"
    except Exception:
        # Intake's safety guarantee is more important than a best-effort
        # normalisation collaborator.  No exception text is retained or logged.
        intake_reason = "invalid_phone"

    steps.append(
        StepResult(
            name="intake",
            status="succeeded",
            reason=intake_reason,
            duration_ms=_duration_ms(intake_started_at),
        )
    )

    score_started_at = time.perf_counter()
    fresh_score: typing.Any | None = None
    try:
        fresh_score = active_deps.score_lead(intake_lead)
    except Exception as exc:
        error, retryable = _classify(exc, "score", "score")
        steps.append(
            _failed_step(
                canonical_lead_id,
                "score",
                error.category,
                error,
                retryable,
                score_started_at,
                error_name=type(exc).__name__,
            )
        )
        steps.extend(
            [
                _skipped_step("store", "upstream_failed", time.perf_counter()),
                _skipped_step("email", "upstream_failed", time.perf_counter()),
                _skipped_step("sms", "upstream_failed", time.perf_counter()),
                _skipped_step("crm", "upstream_failed", time.perf_counter()),
            ]
        )
        return _build_pipeline_result(
            canonical_lead_id,
            idempotency_key_generated,
            steps,
            created=None,
            stored=None,
            fresh_score=None,
            repair=repair,
            failed_step="score",
        )

    steps.append(
        StepResult(
            name="score",
            status="succeeded",
            reason="scored",
            duration_ms=_duration_ms(score_started_at),
        )
    )

    store_started_at = time.perf_counter()
    try:
        record = LeadRecord(id=canonical_lead_id, lead=intake_lead, score=fresh_score)
        stored, created = active_deps.store_lead(record)
    except Exception as exc:
        error, retryable = _classify(exc, "store", "store")
        steps.append(
            _failed_step(
                canonical_lead_id,
                "store",
                error.category,
                error,
                retryable,
                store_started_at,
                error_name=type(exc).__name__,
            )
        )
        steps.extend(
            [
                _skipped_step("email", "upstream_failed", time.perf_counter()),
                _skipped_step("sms", "upstream_failed", time.perf_counter()),
                _skipped_step("crm", "upstream_failed", time.perf_counter()),
            ]
        )
        return _build_pipeline_result(
            canonical_lead_id,
            idempotency_key_generated,
            steps,
            created=None,
            stored=None,
            fresh_score=fresh_score,
            repair=repair,
            failed_step="store",
        )

    steps.append(
        StepResult(
            name="store",
            status="succeeded",
            reason="created" if created else "existing",
            duration_ms=_duration_ms(store_started_at),
        )
    )

    email_result, stored, email_repair = _run_outbound_step(
        "email",
        stored,
        stored.score.category == "cold",
        active_deps,
        precheck=lambda: None,
        prepare=lambda: active_deps.generate_email(stored.lead, stored.score),
        send=active_deps.send_email,
        prepare_stage="generate",
    )
    steps.append(email_result)
    if email_repair is not None:
        repair.append(email_repair)

    sms_phone: list[str | None] = [None]

    def sms_precheck() -> str | None:
        """Re-normalise the stored phone so retries see new configuration."""
        phone = active_deps.normalize_phone(stored.lead.phone)
        if phone.e164 is not None:
            sms_phone[0] = phone.e164
            return None
        if phone.reason == "needs_country_code":
            return "needs_country_code"
        return "invalid_phone"

    def prepare_sms() -> typing.Any:
        """Build an SMS payload from the current authoritative stored record."""
        return active_deps.generate_sms(
            stored.lead.model_copy(update={"phone": sms_phone[0]}),
            stored.score,
        )

    sms_result, stored, sms_repair = _run_outbound_step(
        "sms",
        stored,
        stored.score.category == "cold",
        active_deps,
        precheck=sms_precheck,
        prepare=prepare_sms,
        send=active_deps.send_sms,
        prepare_stage="generate",
    )
    steps.append(sms_result)
    if sms_repair is not None:
        repair.append(sms_repair)

    crm_result, stored, crm_repair = _run_outbound_step(
        "crm",
        stored,
        stored.score.category == "cold",
        active_deps,
        precheck=lambda: None,
        prepare=lambda: active_deps.contact_from_record(stored),
        send=active_deps.upsert_contact,
        prepare_stage="prepare",
    )
    steps.append(crm_result)
    if crm_repair is not None:
        repair.append(crm_repair)

    return _build_pipeline_result(
        canonical_lead_id,
        idempotency_key_generated,
        steps,
        created=created,
        stored=stored,
        fresh_score=fresh_score,
        repair=repair,
        failed_step=None,
    )


def _build_pipeline_result(
    lead_id: str,
    idempotency_key_generated: bool,
    steps: list[StepResult],
    *,
    created: bool | None,
    stored: LeadRecord | None,
    fresh_score: typing.Any | None,
    repair: list[dict[str, typing.Any]],
    failed_step: str | None,
) -> PipelineResult:
    """Aggregate step outcomes into the stable public pipeline report."""
    persisted = any(step.name == "store" and step.status == "succeeded" for step in steps)
    has_failed_step = any(step.status == "failed" for step in steps)
    if not persisted:
        status: typing.Literal["completed", "partial", "failed"] = "failed"
    elif has_failed_step:
        status = "partial"
    else:
        status = "completed"

    retry_steps = [
        step.name
        for step in steps
        if step.status == "failed" and step.retryable
    ]

    if stored is not None:
        category = stored.score.category
        score = stored.score.score
    elif fresh_score is not None:
        category = fresh_score.category
        score = fresh_score.score
    else:
        category = None
        score = None

    return PipelineResult(
        lead_id=lead_id,
        status=status,
        persisted=persisted,
        created=created,
        idempotency_key_generated=idempotency_key_generated,
        category=category,
        score=score,
        steps=steps,
        retry_steps=retry_steps,
        repair=repair,
        failed_step=failed_step,
    )


# Backwards-compatible public alias used by API routes and job workers.
process_lead = run_pipeline
