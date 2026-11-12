"""
services/llm_service.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI SERVICE — POWERED BY OPENROUTER (NOT OPENAI DIRECTLY)
━━━━━━━━━━━━━━━━═══════════════════════════════════════════════

This service uses OpenRouter (https://openrouter.ai) as the
AI gateway instead of calling the OpenAI API directly.

WHY OPENROUTER?
    OpenRouter is a unified API gateway that routes requests
    to hundreds of AI models (OpenAI, Mistral, Anthropic, etc.)
    from a single endpoint. It offers free-tier models that are
    suitable for development and lower-cost production use cases.

SDK COMPATIBILITY:
    OpenRouter is 100% compatible with the OpenAI Python SDK.
    Only the client initialisation changes (api_key + base_url +
    default_headers). All method calls — chat.completions.create(),
    message structure, response parsing — are identical.

MODEL IN USE:
    The model identifier is read from OPENROUTER_MODEL in .env at call time.
    When unset or blank, the free router "openrouter/free" is used (it picks
    a random available free model, so output quality can vary; set
    OPENROUTER_MODEL for consistent results).

BOOKING LINKS (Phase 8):
    Booking links, recipient addresses, phone numbers and the SMS opt-out are
    added by code, never echoed by the model. The model writes prose only; it
    marks the link position with a placeholder token (emails) or a lead-in phrase
    (SMS). services/calendly_service.py builds the personalised URL, strips any
    URL the model invents, and guarantees the SMS stays within 160 characters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os

from openai import OpenAI

from models import EmailPayload, LeadInput, LeadScore, SMSPayload
from models.lead import HOT_SCORE_MIN, MAX_SCORE, MIN_SCORE, WARM_SCORE_MIN
from services.calendly_service import (
    LINK_TOKEN,
    build_booking_link,
    finish_sms,
    inject_link_into_email_body,
    plan_sms,
)


# ---------------------------------------------------------------------------
# Lazy Loader — _get_client()
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    """
    Initialise and return the OpenRouter client at call time, not import time.
    This ensures load_dotenv() has already populated the environment before
    os.getenv() is called.

    Returns:
        A fully configured OpenAI client instance pointed at OpenRouter.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is not found in the environment.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing or empty. Ensure load_dotenv() is "
            "called in main.py and the key is defined in your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            # OpenRouter uses the Referer to identify the calling application.
            "HTTP-Referer": "http://localhost:8000",
            # Human-readable name shown in the OpenRouter dashboard.
            "X-Title": "Lead Management System",
        },
    )


def _get_model() -> str:
    """
    Return the OpenRouter model identifier to use, read at call time.

    Reads OPENROUTER_MODEL from the environment inside the function so that
    load_dotenv() in main.py has already run. If the variable is unset or
    blank, returns _DEFAULT_MODEL. Otherwise returns the stripped value.
    Never raises and never logs the value.

    Returns:
        The stripped model identifier string to pass to OpenRouter.

    Raises:
        None.
    """
    raw = os.getenv("OPENROUTER_MODEL")
    if not raw or not raw.strip():
        return _DEFAULT_MODEL
    return raw.strip()


# Fallback router that selects an available free model when OPENROUTER_MODEL is unset.
_DEFAULT_MODEL = "openrouter/free"

# Honesty rule shared by the email and SMS prompts.
_HONESTY_RULE = (
    "Do not claim that you have already found, selected, curated, shortlisted "
    "or prepared any homes or listings, and do not describe anything the agent "
    "has already done or will personally do beyond following up, answering "
    "questions and arranging a viewing. Use only the details given above: never "
    "invent prices, addresses, availability or neighbourhood facts."
    " Never call the person a 'lead' and never mention their score, category or priority."
)


# ---------------------------------------------------------------------------
# FUNCTION 1 — score_lead
# ---------------------------------------------------------------------------

def score_lead(lead: LeadInput) -> LeadScore:
    """
    Analyse a LeadInput and return a structured LeadScore.

    Sends a carefully engineered prompt to the AI model and parses the
    JSON response into a validated LeadScore object. A low temperature
    (0.3) is used to keep scoring deterministic and consistent. The lead's
    email address and phone number are deliberately NOT sent to the
    model: scoring does not need them.

    Args:
        lead: A validated LeadInput instance containing the raw lead data.

    Returns:
        A LeadScore instance with score, category, reasoning,
        suggested_action, and priority_level populated.

    Raises:
        json.JSONDecodeError: If the model returns malformed JSON.
        ValidationError: If the parsed JSON fails LeadScore validation.
        openai.APIError: On any underlying OpenRouter / network error.
    """
    client = _get_client()

    prompt = f"""You are an expert real estate lead qualification specialist.
Analyse the following lead and return a structured quality score.

Lead Details:
- Name: {lead.name}
- Property type: {lead.property_type or "Not provided"}
- Preferred location: {lead.preferred_location or "Not provided"}
- Budget: {lead.budget or "Not provided"}
- Timeline: {lead.timeline or "Not provided"}
- Message: {lead.message}

Scoring criteria:
- Property type and preferred-location specificity
- Budget clarity and fit for the requested property
- Timeline urgency (immediate > weeks > months > vague)
- Whether the lead appears to be buying, selling, or renting
- Message clarity and overall intent signals

Category ranges (derive the category from the numeric score exactly):
- hot: {HOT_SCORE_MIN}-{MAX_SCORE}
- warm: {WARM_SCORE_MIN}-{HOT_SCORE_MIN - 1}
- cold: {MIN_SCORE}-{WARM_SCORE_MIN - 1}

Respond ONLY with a valid JSON object — no markdown, no explanation, no
code fences. The JSON must have exactly these keys:

{{
  "score": <integer {MIN_SCORE}-{MAX_SCORE}>,
  "category": <"hot" | "warm" | "cold">,
  "reasoning": "<one concise paragraph explaining the score>",
  "suggested_action": "<recommended next step for the real estate agent>",
  "priority_level": <"urgent" | "normal" | "low">
}}"""

    response = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-only API. You never output anything "
                    "other than valid JSON. No markdown. No prose. No code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # Low temperature for deterministic, reproducible scoring
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown code fences that some models add despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    return LeadScore(**data)


# ---------------------------------------------------------------------------
# FUNCTION 2 — generate_email
# ---------------------------------------------------------------------------

def generate_email(lead: LeadInput, score: LeadScore) -> EmailPayload:
    """
    Generate a personalised follow-up email body for a scored lead.

    Uses the AI model to craft a professional, context-aware email that
    references the lead's specific requirements. A higher temperature (0.7)
    allows for more natural, varied language in the generated copy.

    Args:
        lead:  The original LeadInput containing the lead's details.
        score: The LeadScore produced by score_lead(); used to tailor
               tone and urgency of the email.

    Returns:
        An EmailPayload ready to be handed to gmail_service.send_email().

    Raises:
        json.JSONDecodeError: If the model returns malformed JSON.
        ValueError: If the model did not return a usable subject and body.
        ValidationError: If the parsed JSON fails EmailPayload validation.
        openai.APIError: On any underlying OpenRouter / network error.
    """
    client = _get_client()
    link = build_booking_link(lead, channel="email")

    if link:
        booking_rule = (
            "A personalised booking link is inserted automatically. Write "
            "ONE short call-to-action sentence inviting the lead to book a viewing or "
            "consultation, then put the exact token " + LINK_TOKEN + " on its own line "
            "right after that sentence. Never write any URL or web address yourself."
        )
    else:
        booking_rule = (
            "No booking link is available. Invite the lead to reply to "
            "this email to arrange a viewing or consultation. Do not mention a link and "
            "never write any URL or web address."
        )

    prompt = f"""You are a professional real estate sales copywriter.
Write a personalised follow-up email for the following lead.

Lead Details:
- Name: {lead.name}
- Property type: {lead.property_type or "Not provided"}
- Preferred location: {lead.preferred_location or "Not provided"}
- Budget: {lead.budget or "Not provided"}
- Timeline: {lead.timeline or "Not provided"}
- Message: {lead.message}

Lead Score: {score.score}/100 (Category: {score.category})
Suggested Action: {score.suggested_action}

Email guidelines:
- Address the lead by first name only
- Be warm, professional, and concise (3–4 short paragraphs)
- Tailor the reply to the lead's apparent buying, selling, or renting need
- {_HONESTY_RULE}
- Reference their property type, preferred location, and budget naturally when provided
- {booking_rule}
- Do NOT use generic filler phrases like "I hope this email finds you well"
- Tone: {"urgent and enthusiastic" if score.category == "hot" else "helpful and professional for a home buyer, seller, or renter"}

Respond ONLY with a valid JSON object — no markdown, no explanation:

{{
  "subject": "<compelling email subject line>",
  "body": "<full email body as a single string with \\n for line breaks>"
}}"""

    response = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-only API. You never output anything "
                    "other than valid JSON. No markdown. No prose. No code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,  # Higher temperature for more natural, varied email copy
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("subject"), str)
        or not data["subject"].strip()
        or not isinstance(data.get("body"), str)
        or not data["body"].strip()
    ):
        raise ValueError("The model did not return a usable subject and body.")

    subject = data["subject"].strip()
    body = inject_link_into_email_body(data["body"], link)

    return EmailPayload(
        to_email=lead.email,
        to_name=lead.name,
        subject=subject,
        body=body,
        lead_score=score.score,
        calendly_link=link,
    )


# ---------------------------------------------------------------------------
# FUNCTION 3 — generate_sms
# ---------------------------------------------------------------------------

def generate_sms(lead: LeadInput, score: LeadScore) -> SMSPayload:
    """
    Generate a concise SMS follow-up message for a scored lead.

    Produces a single SMS message within the 160-character GSM-7 limit.
    A moderate temperature (0.7) allows for natural phrasing while keeping
    the message tight and within the hard character cap enforced by SMSPayload.

    Args:
        lead:  The original LeadInput containing the lead's contact details.
        score: The LeadScore produced by score_lead(); used to set urgency.

    Returns:
        An SMSPayload ready to be handed to twilio_service.send_sms().

    Raises:
        json.JSONDecodeError: If the model returns malformed JSON.
        ValueError: If the model did not return a usable message.
        ValidationError: If the parsed JSON fails SMSPayload validation
                         (e.g., message exceeds 160 characters).
        openai.APIError: On any underlying OpenRouter / network error.
    """
    client = _get_client()
    plan = plan_sms(lead)

    if plan.link:
        booking_rule = (
            "End the text with a short phrase that leads into a booking link, "
            "such as 'Book a viewing here:'. The link itself is appended automatically."
        )
    else:
        booking_rule = "Do not mention booking links or scheduling."

    prompt = f"""You are a real estate SMS copywriter.
Write a short, professional SMS follow-up for the following lead.

Lead Details:
- Name: {lead.name} (use first name only)
- Property type: {lead.property_type or "Not specified"}
- Preferred location: {lead.preferred_location or "Not specified"}
- Budget: {lead.budget or "Not specified"}
- Timeline: {lead.timeline or "Not specified"}

Lead Score: {score.score}/100 (Category: {score.category})

SMS guidelines:
- CRITICAL: The message MUST be {plan.text_budget} characters or fewer (including spaces)
- Be warm, professional, and direct
- Tailor the reply to the lead's apparent buying, selling, or renting need
- {_HONESTY_RULE}
- Mention that an agent will be in touch shortly
- Do NOT include any URL and do NOT include an opt-out or STOP instruction: both are added automatically.
- {booking_rule}
- Count characters carefully before responding

Respond ONLY with a valid JSON object — no markdown, no explanation:

{{
  "message": "<SMS text, max {plan.text_budget} characters>"
}}"""

    response = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise JSON-only API. You never output anything "
                    "other than valid JSON. No markdown. No prose. No code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,  # Higher temperature for natural, varied SMS copy
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    raw_message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(raw_message, str) or not raw_message.strip():
        raise ValueError("The model did not return a usable message.")

    message = finish_sms(raw_message, plan, lead)

    return SMSPayload(
        to_phone=lead.phone,
        message=message,
        lead_score=score.score,
    )
