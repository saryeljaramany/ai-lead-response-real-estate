"""
models/lead.py

Core data contracts for the AI-powered lead management system.
Every service in this application — AI scoring, database persistence,
CRM integration, and email/SMS dispatch — depends on the three models
defined here. Treat them as the single source of truth for all lead data.
"""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator, model_validator


MIN_SCORE = 0
WARM_SCORE_MIN = 40
HOT_SCORE_MIN = 70
MAX_SCORE = 100


# ---------------------------------------------------------------------------
# MODEL 1 — LeadInput
# ---------------------------------------------------------------------------

class LeadInput(BaseModel):
    """
    Represents raw lead data submitted into the system.

    This is the entry point for all inbound lead data, whether it arrives
    via the REST API, a web form, or a third-party integration. It performs
    basic sanitisation and validation before the data is handed off to any
    downstream service (AI scoring, CRM, etc.).

    Role in the system:
        - Received by POST /leads as the request body.
        - Embedded inside LeadRecord after scoring.
        - Never mutated after creation; treat as immutable downstream.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Sarah Mitchell",
                "email": "sarah.mitchell@example.com",
                "phone": "+1-415-555-0192",
                "property_type": "house",
                "preferred_location": "Austin, TX",
                "budget": "$450k-$550k",
                "timeline": "within 3 months",
                "message": (
                    "My partner and I are looking for a three-bedroom home near "
                    "good schools and would like to move before summer."
                ),
            }
        }
    )

    # Required fields
    name: str                       # Full name of the lead/contact person
    email: EmailStr                 # Validated email address (pydantic EmailStr)
    phone: str                      # Contact phone number (any international format)
    message: str                    # Free-text enquiry or message from the lead

    # Optional fields — may not be present for all inbound channels
    property_type: str | None = None  # Desired property, e.g. house, apartment, condo, or land
    preferred_location: str | None = None  # Desired city, neighbourhood, or area
    budget: str | None = None       # Budget range as a human-readable string, e.g. "$450k-$550k"
    timeline: str | None = None     # Purchase/decision timeline, e.g. "immediate", "3 months"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Strip surrounding whitespace; reject if the result is empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("name must not be empty or whitespace only")
        return stripped

    @field_validator("phone", mode="before")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        """
        Strip surrounding whitespace; enforce a minimum length of 7 characters
        to filter out obviously invalid numbers while remaining format-agnostic
        (supports local, national, and E.164 international formats).
        """
        stripped = v.strip()
        if len(stripped) < 7:
            raise ValueError(
                "phone must be at least 7 characters after stripping whitespace"
            )
        return stripped

    @field_validator("message", mode="before")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Strip surrounding whitespace; reject if the result is empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace only")
        return stripped


# ---------------------------------------------------------------------------
# MODEL 2 — LeadScore
# ---------------------------------------------------------------------------

class LeadScore(BaseModel):
    """
    Represents the structured output returned by the AI scoring service.

    After a LeadInput is received, it is sent to the AI scoring pipeline
    which analyses the content and returns a LeadScore. This model enforces
    strict constraints on the AI output so that downstream services (CRM
    routing, email dispatch, priority queues) can rely on consistent,
    normalised values.

    Role in the system:
        - Produced exclusively by services/ai_scorer.py.
        - Embedded inside LeadRecord alongside the originating LeadInput.
        - Consumed by routers and CRM integration to drive automated actions.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "score": 87,
                "category": "hot",
                "reasoning": (
                    "Lead has a clear budget ($450k-$550k), a preferred Austin "
                    "location, and a concrete three-month home-buying timeline."
                ),
                "suggested_action": (
                    "Contact the lead today and offer a property viewing this week."
                ),
                "priority_level": "urgent",
            }
        }
    )

    score: int              # Numeric quality score in the range [0, 100]
    category: str | None = "cold"  # Derived lead temperature: "hot" | "warm" | "cold"
    reasoning: str          # Human-readable explanation of the score from the AI
    suggested_action: str   # Recommended next step for the real estate agent
    priority_level: str     # Operational priority: "urgent" | "normal" | "low"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: int) -> int:
        """Ensure score is within the valid [0, 100] range (inclusive)."""
        if not (MIN_SCORE <= v <= MAX_SCORE):
            raise ValueError(
                f"score must be between {MIN_SCORE} and {MAX_SCORE} inclusive, got {v}"
            )
        return v

    @model_validator(mode="after")
    def derive_category(self) -> "LeadScore":
        """Derive the lead temperature from the validated numeric score."""
        if self.score >= HOT_SCORE_MIN:
            self.category = "hot"
        elif self.score >= WARM_SCORE_MIN:
            self.category = "warm"
        else:
            self.category = "cold"
        return self

    @field_validator("priority_level", mode="before")
    @classmethod
    def validate_priority_level(cls, v: str) -> str:
        """
        Normalise to lowercase and restrict to allowed priority levels.
        Accepts case-insensitive input and always stores as lowercase.
        """
        normalised = v.strip().lower()
        allowed = {"urgent", "normal", "low"}
        if normalised not in allowed:
            raise ValueError(
                f"priority_level must be one of {allowed}, got '{v}'"
            )
        return normalised


# ---------------------------------------------------------------------------
# MODEL 3 — LeadRecord
# ---------------------------------------------------------------------------

class LeadRecord(BaseModel):
    """
    Represents the complete lead object persisted to the database.

    LeadRecord is the canonical, long-lived representation of a lead after
    it has been submitted and scored. It composes LeadInput and LeadScore
    with system-level metadata and tracks the state of every automated
    action taken on the lead (email, SMS, HubSpot sync).

    Role in the system:
        - Written to the database by services/storage.py immediately after
          AI scoring completes.
        - Returned by GET /leads/{id} as the authoritative lead view.
        - Read by services/hubspot.py, services/email.py, and
          services/sms.py to dispatch communications idempotently.
        - The boolean flags (email_sent, sms_sent, hubspot_created) act
          as idempotency guards; services must check these before acting.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "lead": {
                    "name": "Sarah Mitchell",
                    "email": "sarah.mitchell@example.com",
                    "phone": "+1-415-555-0192",
                    "property_type": "house",
                    "preferred_location": "Austin, TX",
                    "budget": "$450k-$550k",
                    "timeline": "within 3 months",
                    "message": (
                        "My partner and I are looking for a three-bedroom home "
                        "near good schools and would like to move before summer."
                    ),
                },
                "score": {
                    "score": 87,
                    "category": "hot",
                    "reasoning": (
                        "Lead has a clear budget, preferred location, and a "
                        "concrete home-buying timeline."
                    ),
                    "suggested_action": (
                        "Contact the lead today and offer a property viewing this week."
                    ),
                    "priority_level": "urgent",
                },
                "email_sent": False,
                "sms_sent": False,
                "hubspot_created": False,
                "created_at": "2026-04-25T21:00:00+00:00",
            }
        }
    )

    # System-generated unique identifier for this record
    id: str = ""                    # UUID4 string; auto-generated via default_factory

    # Core nested models
    lead: LeadInput                 # The original sanitised input data for this lead
    score: LeadScore                # The AI-generated quality score and metadata

    # Communication dispatch flags — used as idempotency guards
    email_sent: bool = False        # True once a follow-up email has been sent successfully
    sms_sent: bool = False          # True once a follow-up SMS has been sent successfully
    hubspot_created: bool = False   # True once the lead has been created in HubSpot CRM

    # Audit timestamp
    created_at: datetime | None = None  # UTC timestamp of when this record was first persisted

    def model_post_init(self, __context) -> None:
        """Apply default_factory logic for fields that require callable defaults."""
        if not self.id:
            # Cast uuid4() to str so the field remains a plain string,
            # compatible with JSON serialisation and all database drivers.
            object.__setattr__(self, "id", str(uuid4()))
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc))
