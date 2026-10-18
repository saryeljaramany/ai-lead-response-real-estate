"""
models/__init__.py

Centralised export point for all Pydantic models in the lead management system.
This file allows for clean imports throughout the application, e.g.:
    from models import LeadInput, EmailPayload

Available Models:
    - LeadInput: Raw lead data entry point.
    - LeadScore: AI-generated scoring and category data.
    - LeadRecord: Complete database persistence record.
    - EmailPayload: Data contract for the Gmail service.
    - SMSPayload: Data contract for the Twilio service.
    - HubSpotContact: Data contract for the HubSpot CRM service.
"""

from .lead import (
    LeadInput,   # Used by API routers for inbound lead submission
    LeadScore,   # Used by AI service for structured scoring output
    LeadRecord,  # Used by storage service for database persistence
)
from .communication import (
    EmailPayload,    # Used by gmail_service.py for email dispatch
    SMSPayload,      # Used by twilio_service.py for SMS dispatch
    HubSpotContact,  # Used by hubspot_service.py for CRM synchronization
)

__all__ = [
    "LeadInput",
    "LeadScore",
    "LeadRecord",
    "EmailPayload",
    "SMSPayload",
    "HubSpotContact",
]
