from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import SMSPayload
from services.twilio_service import SMSSendError, send_sms

router = APIRouter(prefix="/sms", tags=["SMS"])


class SMSSendResponse(BaseModel):
    """HTTP-layer response model for a successful SMS send attempt."""

    message: str
    to: str
    message_id: str | None
    status: str
    dry_run: bool
    encoding: str
    segments_estimated: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "SMS accepted for delivery.",
                    "to": "+14155550123",
                    "message_id": "SM1234567890abcdef1234567890abcdef",
                    "status": "queued",
                    "dry_run": True,
                    "encoding": "GSM-7",
                    "segments_estimated": 1,
                }
            ]
        }
    }


@router.post(
    "/send",
    response_model=SMSSendResponse,
    responses={
        200: {
            "description": (
                "SMS accepted (status='queued') or simulated (dry_run=True). "
                "'queued' means the message was accepted by Twilio, not that it "
                "has been delivered. When dry_run=True no message was sent."
            ),
            "model": SMSSendResponse,
        },
        422: {
            "description": (
                "Validation error or undeliverable recipient. "
                "The request must be corrected before retrying — retrying as-is "
                "will not succeed."
            ),
            "content": {
                "application/json": {
                    "example": {"detail": "The 'to' number is not a valid phone number."}
                }
            },
        },
        500: {
            "description": (
                "Unexpected server-side error or Twilio configuration problem. "
                "Not retryable without investigation."
            ),
            "content": {
                "application/json": {
                    "example": {"detail": "Unexpected error while sending the SMS."}
                }
            },
        },
        502: {
            "description": (
                "Transient upstream error from Twilio. "
                "The request MAY be retried, but be aware that the message could "
                "have already been queued by Twilio before the error was raised, "
                "so retrying carries a risk of duplicate delivery."
            ),
            "content": {
                "application/json": {
                    "example": {"detail": "Twilio service temporarily unavailable. Please retry."}
                }
            },
        },
    },
)
def send_sms_endpoint(payload: SMSPayload) -> SMSSendResponse:
    """Send an SMS message via Twilio.

    Accepts an :class:`~models.SMSPayload` request body and delegates to
    :func:`~services.twilio_service.send_sms` which owns all Twilio
    interaction logic.  ``to_phone`` is normalised to strict E.164 format by
    the service, and dry-run mode is controlled solely by the
    ``TWILIO_DRY_RUN`` environment variable.

    **What it receives**
        A JSON body conforming to ``SMSPayload`` with required fields:

        * ``to_phone`` – recipient phone number (normalised to E.164 by
          :func:`~services.twilio_service.send_sms`).
        * ``message`` – SMS body text (1–160 characters).
        * ``lead_score`` – integer quality score in ``[0, 100]``.

        There are no per-request ``dry_run`` or ``from_`` options; dry-run
        behaviour is governed exclusively by ``TWILIO_DRY_RUN``.

    **What it returns**
        An :class:`SMSSendResponse` with the following notable fields:

        * ``status`` – ``"queued"`` means Twilio has *accepted* the message;
          it does **not** guarantee delivery.  Poll the Twilio webhook or status
          callback URL for final delivery confirmation.
        * ``dry_run`` – when ``True`` the endpoint went through all validation
          steps but **no message was transmitted**; Twilio was never called.
        * ``message_id`` – the Twilio message SID, or ``None`` for dry runs.
        * ``encoding`` – character encoding detected for the message body
          (e.g. ``"GSM-7"`` or ``"UCS-2"``).
        * ``segments_estimated`` – estimated number of SMS segments that will
          be billed.

    **Errors**

    * **422 Unprocessable Entity** – returned for two distinct reasons:
        1. Pydantic validation failure on the request body (FastAPI built-in).
        2. The recipient number is invalid or unreachable
           (``SMSSendError.category == "recipient"``), or a ``ValueError`` was
           raised during processing.
        These are *not* retryable without fixing the request.

    * **500 Internal Server Error** – Twilio account misconfiguration
      (``SMSSendError.category == "config"``) or any other unexpected
      exception.  Not retryable without investigation.

    * **502 Bad Gateway** – Transient upstream failure from Twilio
      (``SMSSendError.category == "transient"``).  The request *may* be
      retried after a backoff period, but carry in mind that Twilio may have
      already accepted the message before returning the error, so retrying
      **risks duplicate delivery**.
    """
    try:
        return send_sms(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SMSSendError as exc:
        status = {"recipient": 422, "transient": 502}.get(exc.category, 500)
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected error while sending the SMS.") from exc
