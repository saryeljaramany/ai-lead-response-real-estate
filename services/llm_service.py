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
    "mistralai/mistral-7b-instruct:free"
    A free-tier instruction-tuned model on OpenRouter. It handles
    structured JSON output reliably when prompted explicitly to
    respond only with valid JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os

from openai import OpenAI

from models import EmailPayload, LeadInput, LeadScore, SMSPayload
from models.lead import HOT_SCORE_MIN, MAX_SCORE, MIN_SCORE, WARM_SCORE_MIN


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


# The free-tier model used for all three AI functions in this service.
_MODEL = "mistralai/mistral-7b-instruct:free"


# ---------------------------------------------------------------------------
# FUNCTION 1 — score_lead
# ---------------------------------------------------------------------------

def score_lead(lead: LeadInput) -> LeadScore:
    """
    Analyse a LeadInput and return a structured LeadScore.

    Sends a carefully engineered prompt to the AI model and parses the
    JSON response into a validated LeadScore object. A low temperature
    (0.3) is used to keep scoring deterministic and consistent.

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
- Email: {lead.email}
- Phone: {lead.phone}
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
        model=_MODEL,
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
        ValidationError: If the parsed JSON fails EmailPayload validation.
        openai.APIError: On any underlying OpenRouter / network error.
    """
    client = _get_client()
    calendly_link = os.getenv("CALENDLY_LINK", "")

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
{"Calendly Booking Link: " + calendly_link if calendly_link else ""}

Email guidelines:
- Address the lead by first name only
- Be warm, professional, and concise (3–4 short paragraphs)
- Tailor the reply to the lead's apparent buying, selling, or renting need
- Reference their property type, preferred location, and budget naturally when provided
- If a Calendly link is provided, include a clear call-to-action for a property viewing or consultation
- Do NOT use generic filler phrases like "I hope this email finds you well"
- Tone: {"urgent and enthusiastic for a ready-to-act home lead" if score.category == "hot" else "helpful and professional for a home buyer, seller, or renter"}

Respond ONLY with a valid JSON object — no markdown, no explanation:

{{
  "to_email": "{lead.email}",
  "to_name": "{lead.name}",
  "subject": "<compelling email subject line>",
  "body": "<full email body as a single string with \\n for line breaks>",
  "lead_score": {score.score},
  "calendly_link": {"\"" + calendly_link + "\"" if calendly_link else "null"}
}}"""

    response = client.chat.completions.create(
        model=_MODEL,
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
    return EmailPayload(**data)


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
        ValidationError: If the parsed JSON fails SMSPayload validation
                         (e.g., message exceeds 160 characters).
        openai.APIError: On any underlying OpenRouter / network error.
    """
    client = _get_client()

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
- CRITICAL: The message MUST be 160 characters or fewer (including spaces)
- Be warm, professional, and direct
- Tailor the reply to the lead's apparent buying, selling, or renting need
- Mention that a real estate agent will be in touch shortly
- Include an opt-out instruction: "Reply STOP to opt out"
- Count characters carefully before responding

Respond ONLY with a valid JSON object — no markdown, no explanation:

{{
  "to_phone": "{lead.phone}",
  "message": "<SMS body, max 160 characters>",
  "lead_score": {score.score}
}}"""

    response = client.chat.completions.create(
        model=_MODEL,
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

    # Hard-truncate the message as a last-resort safety net before Pydantic
    # validation, in case the model slightly overshoots the 160-char limit
    # despite explicit instructions. The validator will still raise if the
    # truncated result is empty, preserving the model's integrity.
    if "message" in data and len(data["message"]) > 160:
        data["message"] = data["message"][:157] + "..."

    return SMSPayload(**data)
