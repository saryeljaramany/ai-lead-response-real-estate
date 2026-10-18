"""
models/communication.py

Outbound communication and CRM data contracts for the AI-powered
lead management system.

This file is intentionally independent of models/lead.py so that each
external-service adapter (Gmail, Twilio, HubSpot) can import only what
it needs without pulling in the full lead graph.

Service mapping:
    EmailPayload    → consumed by services/gmail_service.py
    SMSPayload      → consumed by services/twilio_service.py
    HubSpotContact  → consumed by services/hubspot_service.py
"""

from pydantic import BaseModel, EmailStr, field_validator, ConfigDict


# ---------------------------------------------------------------------------
# MODEL 1 — EmailPayload
# ---------------------------------------------------------------------------

class EmailPayload(BaseModel):
    """
    Represents all data required to compose and send a single email to a lead.

    Passed directly into gmail_service.py after AI scoring is complete. The
    service uses these fields to build the MIME message, authenticate with the
    Gmail API, and dispatch the email. The optional calendly_link field is
    injected into the body template when a booking link should be included.

    Role in the system:
        - Constructed in the lead processing pipeline once a LeadScore exists.
        - Handed to gmail_service.send_email(payload: EmailPayload).
        - lead_score is not sent to the recipient; it is attached to service
          logs and audit records for traceability.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "to_email": "sarah.mitchell@example.com",
                "to_name": "Sarah Mitchell",
                "subject": "Your Austin Home Search",
                "body": (
                    "Hi Sarah,\n\nThank you for reaching out. Based on your "
                    "home search, we have selected properties that may be a "
                    "great fit. Please use the link below to book a property "
                    "viewing.\n\nBest regards,\n"
                    "The Real Estate Team"
                ),
                "lead_score": 87,
                "calendly_link": "https://calendly.com/realestate-team/property-viewing",
            }
        }
    )

    to_email: EmailStr          # Recipient email address — validated by pydantic EmailStr
    to_name: str                # Recipient display name used in the email greeting
    subject: str                # Email subject line
    body: str                   # Full plain-text (or HTML) body of the email
    lead_score: int             # AI quality score [0-100]; used for logging, not sent to lead
    calendly_link: str | None = None  # Optional booking URL injected into the body template

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("subject", mode="before")
    @classmethod
    def validate_subject(cls, v: str) -> str:
        """Strip surrounding whitespace; reject if the result is empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("subject must not be empty or whitespace only")
        return stripped

    @field_validator("body", mode="before")
    @classmethod
    def validate_body(cls, v: str) -> str:
        """Strip surrounding whitespace; reject if the result is empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("body must not be empty or whitespace only")
        return stripped

    @field_validator("lead_score")
    @classmethod
    def validate_lead_score(cls, v: int) -> int:
        """Ensure lead_score is within the valid [0, 100] range (inclusive)."""
        if not (0 <= v <= 100):
            raise ValueError(
                f"lead_score must be between 0 and 100 inclusive, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# MODEL 2 — SMSPayload
# ---------------------------------------------------------------------------

class SMSPayload(BaseModel):
    """
    Represents all data required to send a single SMS message to a lead.

    Passed directly into twilio_service.py. The service reads to_phone and
    message to make the Twilio REST API call. The 160-character message cap
    is enforced here rather than silently truncated, so the caller is forced
    to craft concise copy at build-time rather than discovering truncation in
    production logs.

    Role in the system:
        - Constructed in the lead processing pipeline for high-priority leads.
        - Handed to twilio_service.send_sms(payload: SMSPayload).
        - lead_score is used for logging and audit only; not transmitted.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "to_phone": "+14155550192",
                "message": (
                    "Hi Sarah, we found homes that may fit your search. A real "
                    "estate agent will contact you shortly. Reply STOP to opt out."
                ),
                "lead_score": 87,
            }
        }
    )

    to_phone: str       # Destination phone number (E.164 format recommended, e.g. +14155550192)
    message: str        # SMS body — must be non-empty and within the 160-character GSM-7 limit
    lead_score: int     # AI quality score [0-100]; used for logging, not transmitted in the SMS

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("to_phone", mode="before")
    @classmethod
    def validate_to_phone(cls, v: str) -> str:
        """
        Strip surrounding whitespace; enforce a minimum length of 7 characters
        to filter out obviously invalid numbers while remaining format-agnostic.
        """
        stripped = v.strip()
        if len(stripped) < 7:
            raise ValueError(
                "to_phone must be at least 7 characters after stripping whitespace"
            )
        return stripped

    @field_validator("message", mode="before")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """
        Strip surrounding whitespace; reject if empty; enforce the standard
        single-segment SMS limit of 160 characters to prevent silent splitting
        or truncation by the Twilio gateway.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace only")
        if len(stripped) > 160:
            raise ValueError(
                f"message must not exceed 160 characters (got {len(stripped)}). "
                "Shorten the copy or use a multi-part SMS strategy."
            )
        return stripped

    @field_validator("lead_score")
    @classmethod
    def validate_lead_score(cls, v: int) -> int:
        """Ensure lead_score is within the valid [0, 100] range (inclusive)."""
        if not (0 <= v <= 100):
            raise ValueError(
                f"lead_score must be between 0 and 100 inclusive, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# MODEL 3 — HubSpotContact
# ---------------------------------------------------------------------------

class HubSpotContact(BaseModel):
    """
    Represents lead data formatted specifically for the HubSpot CRM API.

    Field names align with HubSpot's standard contact properties so the
    dictionary produced by model.model_dump() can be passed almost directly
    to the HubSpot contacts API endpoint without a separate mapping layer.
    The lead_status field maps to HubSpot's lifecycle stage / lead status
    property and must be one of the system-defined buckets.

    Role in the system:
        - Constructed in the lead processing pipeline after AI scoring.
        - Handed to hubspot_service.create_contact(contact: HubSpotContact).
        - hubspot_created flag on LeadRecord is set to True after a successful
          API call to prevent duplicate contact creation on retries.

    HubSpot API reference:
        POST https://api.hubapi.com/crm/v3/objects/contacts
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "sarah.mitchell@example.com",
                "firstname": "Sarah",
                "lastname": "Mitchell",
                "phone": "+14155550192",
                "property_type": "house",
                "preferred_location": "Austin, TX",
                "lead_score": 87,
                "lead_status": "hot",
            }
        }
    )

    email: EmailStr             # Contact's email — primary unique identifier in HubSpot
    firstname: str              # Contact's first name (maps to HubSpot property: firstname)
    lastname: str               # Contact's last name  (maps to HubSpot property: lastname)
    phone: str                  # Contact's phone number (maps to HubSpot property: phone)
    property_type: str | None = None  # Desired property type; requires a matching HubSpot property
    preferred_location: str | None = None  # Desired area; requires a matching HubSpot property
    lead_score: int             # AI quality score [0-100] (maps to custom HubSpot property)
    lead_status: str            # Lead temperature: "hot" | "warm" | "cold"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("firstname", mode="before")
    @classmethod
    def validate_firstname(cls, v: str) -> str:
        """Strip surrounding whitespace; reject if the result is empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("firstname must not be empty or whitespace only")
        return stripped

    @field_validator("lastname", mode="before")
    @classmethod
    def validate_lastname(cls, v: str) -> str:
        """Strip surrounding whitespace; reject if the result is empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("lastname must not be empty or whitespace only")
        return stripped

    @field_validator("lead_score")
    @classmethod
    def validate_lead_score(cls, v: int) -> int:
        """Ensure lead_score is within the valid [0, 100] range (inclusive)."""
        if not (0 <= v <= 100):
            raise ValueError(
                f"lead_score must be between 0 and 100 inclusive, got {v}"
            )
        return v

    @field_validator("lead_status", mode="before")
    @classmethod
    def validate_lead_status(cls, v: str) -> str:
        """
        Normalise to lowercase and restrict to the three allowed status buckets.
        Accepts case-insensitive input ("Hot", "HOT", etc.) and always stores
        as lowercase so HubSpot property comparisons are consistent.
        """
        normalised = v.strip().lower()
        allowed = {"hot", "warm", "cold"}
        if normalised not in allowed:
            raise ValueError(
                f"lead_status must be one of {allowed}, got '{v}'"
            )
        return normalised
