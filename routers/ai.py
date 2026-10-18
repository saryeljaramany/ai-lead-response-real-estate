"""
routers/ai.py

HTTP routing layer for all AI-powered endpoints in the lead management system.

This file contains zero business logic. Its sole responsibility is to:
    1. Accept and validate HTTP request bodies via Pydantic models.
    2. Delegate work to the appropriate function in services/llm_service.py.
    3. Return the service result as an HTTP response, or map exceptions to
       appropriate HTTP error codes.

Router prefix:  /ai
OpenAPI tag:    AI
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models import EmailPayload, LeadInput, LeadScore, SMSPayload
from services.llm_service import generate_email, generate_sms, score_lead

router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)


# ---------------------------------------------------------------------------
# Request wrapper models
# (defined here rather than in models/ because they are HTTP-layer concerns,
#  not core domain contracts)
# ---------------------------------------------------------------------------

class GenerateEmailRequest(BaseModel):
    """
    Composite request body for the POST /ai/generate-email endpoint.

    Bundles the original lead data together with its AI-generated score so
    the email generation function has all the context it needs to produce
    personalised, appropriately-toned copy in a single call.

    Fields:
        lead:  The sanitised lead data originally submitted to the system.
        score: The LeadScore returned by the POST /ai/score-lead endpoint.
    """

    lead: LeadInput    # Originating lead — used to personalise the email body
    score: LeadScore   # AI score — used to set tone, urgency, and suggested action


class GenerateSMSRequest(BaseModel):
    """
    Composite request body for the POST /ai/generate-sms endpoint.

    Bundles the original lead data together with its AI-generated score so
    the SMS generation function can produce a concise, context-aware message
    within the 160-character GSM-7 limit.

    Fields:
        lead:  The sanitised lead data originally submitted to the system.
        score: The LeadScore returned by the POST /ai/score-lead endpoint.
    """

    lead: LeadInput    # Originating lead — supplies name and phone number
    score: LeadScore   # AI score — used to calibrate urgency and priority


# ===========================================================================
# ENDPOINT 1 — Score Lead
# ===========================================================================

@router.post(
    "/score-lead",
    response_model=LeadScore,
    summary="Score a lead with AI",
)
async def score_lead_endpoint(lead: LeadInput) -> LeadScore:
    """
    Analyse a lead submission and return an AI-generated quality score.

    Passes the inbound LeadInput to the OpenRouter AI service which evaluates
    budget, timeline, message quality, and purchase-intent signals before
    returning a structured LeadScore.

    **Receives:**
    A `LeadInput` JSON body with the following required fields:
    - `name`, `email`, `phone`, `message`

    And the following optional fields:
    - `property_type`, `preferred_location`, `budget`, `timeline`

    **Returns:**
    A `LeadScore` object containing:
    - `score` (0–100 integer)
    - `category` ("hot" | "warm" | "cold")
    - `reasoning` (AI explanation of the score)
    - `suggested_action` (recommended next step)
    - `priority_level` ("urgent" | "normal" | "low")

    **Errors:**
    - `422 Unprocessable Entity` — input validation failed or the AI returned
      an unexpected value that could not be coerced into a valid LeadScore.
    - `500 Internal Server Error` — unexpected error in the AI service or
      network layer (OpenRouter outage, malformed JSON from model, etc.).
    """
    try:
        return score_lead(lead)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while scoring the lead. "
                   "Please try again later.",
        ) from exc


# ===========================================================================
# ENDPOINT 2 — Generate Email
# ===========================================================================

@router.post(
    "/generate-email",
    response_model=EmailPayload,
    summary="Generate a personalised follow-up email for a lead",
)
async def generate_email_endpoint(request: GenerateEmailRequest) -> EmailPayload:
    """
    Generate a personalised follow-up email for a scored lead.

    Passes the lead and its AI score to the OpenRouter AI service, which
    crafts professional, context-aware email copy tailored to the lead's
    requirements and budget. The resulting payload is ready to be forwarded
    directly to `gmail_service.send_email()`.

    **Receives:**
    A `GenerateEmailRequest` JSON body containing:
    - `lead` — the original `LeadInput` object
    - `score` — the `LeadScore` object returned by `/ai/score-lead`

    **Returns:**
    An `EmailPayload` object containing:
    - `to_email`, `to_name` — recipient addressing
    - `subject` — AI-generated subject line
    - `body` — full personalised email body
    - `lead_score` — echoed score for logging/audit
    - `calendly_link` — injected from environment if configured, else `null`

    **Errors:**
    - `422 Unprocessable Entity` — validation failed or the AI returned a
      value that could not be coerced into a valid EmailPayload.
    - `500 Internal Server Error` — unexpected error in the AI service
      (OpenRouter outage, malformed JSON from model, etc.).
    """
    try:
        return generate_email(request.lead, request.score)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the email. "
                   "Please try again later.",
        ) from exc


# ===========================================================================
# ENDPOINT 3 — Generate SMS
# ===========================================================================

@router.post(
    "/generate-sms",
    response_model=SMSPayload,
    summary="Generate a concise SMS follow-up for a lead",
)
async def generate_sms_endpoint(request: GenerateSMSRequest) -> SMSPayload:
    """
    Generate a concise SMS follow-up message for a scored lead.

    Passes the lead and its AI score to the OpenRouter AI service, which
    produces a professional, direct SMS message within the standard 160-
    character GSM-7 single-segment limit. The resulting payload is ready
    to be forwarded directly to `twilio_service.send_sms()`.

    **Receives:**
    A `GenerateSMSRequest` JSON body containing:
    - `lead` — the original `LeadInput` object (phone number is used as recipient)
    - `score` — the `LeadScore` object returned by `/ai/score-lead`

    **Returns:**
    An `SMSPayload` object containing:
    - `to_phone` — destination phone number taken from the lead
    - `message` — AI-generated SMS body (max 160 characters)
    - `lead_score` — echoed score for logging/audit

    **Errors:**
    - `422 Unprocessable Entity` — validation failed or the AI-generated
      message exceeded 160 characters after truncation fallback.
    - `500 Internal Server Error` — unexpected error in the AI service
      (OpenRouter outage, malformed JSON from model, etc.).
    """
    try:
        return generate_sms(request.lead, request.score)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the SMS. "
                   "Please try again later.",
        ) from exc
