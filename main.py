# =============================================================================
# Imports
# =============================================================================
from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from routers import leads, ai, email, sms, crm, captcha
from services.security import require_api_key

# =============================================================================
# Environment Configuration
# =============================================================================
load_dotenv()

APP_TITLE = os.getenv("APP_TITLE", "Real Estate Leads API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "A FastAPI application for managing real estate leads.",
)

# =============================================================================
# Application Instance
# =============================================================================
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    dependencies=[Depends(require_api_key)],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================
_ALLOWED_VALIDATION_ERROR_KEYS = ("loc", "msg", "type")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle request validation errors without echoing raw input or context.

    Purpose:
        Replaces FastAPI's default 422 response payload with a sanitized list
        of error dictionaries containing only ``"loc"``, ``"msg"``, and
        ``"type"`` from each item of ``exc.errors()``, serialized through
        ``fastapi.encoders.jsonable_encoder``.

    Privacy & Serialization Rationale:
        Pydantic v2 validation errors include ``"input"`` (which for a missing
        nested field echoes the entire parent object, exposing lead PII such
        as name, email, phone, and message), ``"ctx"`` (which can hold raw
        values or non-JSON-serialisable objects such as ``ValueError``
        instances), and ``"url"``. Whitelisting only ``"loc"``, ``"msg"``, and
        ``"type"`` prevents lead PII echo and serialization errors. Note that
        ``"msg"`` can still quote a short non-PII value (for example
        ``got 'lukewarm'``); only ``input``, ``ctx``, ``url``, and any other
        keys are removed.

    Returns:
        JSONResponse with status code 422 and ``{"detail": <sanitized list>}``.
    """
    sanitized = [
        {key: error[key] for key in _ALLOWED_VALIDATION_ERROR_KEYS if key in error}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(sanitized)},
    )


# =============================================================================
# REGISTER ROUTERS
# =============================================================================
app.include_router(leads.router)  # handles /leads routes
app.include_router(ai.router)     # handles /ai routes
app.include_router(email.router)  # handles /email routes
app.include_router(sms.router)  # handles /sms routes
app.include_router(crm.router)  # handles /crm routes
app.include_router(captcha.router)  # handles /captcha routes

# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.

    Returns the current status of the application along with
    the app name and version, useful for monitoring and
    readiness probes.
    """
    return {
        "status": "ok",
        "app_name": APP_TITLE,
        "version": APP_VERSION,
    }


@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint.

    Returns a welcome message and points clients to the
    interactive API docs (/docs) and the health check
    endpoint (/health).
    """
    return {
        "message": f"Welcome to the {APP_TITLE}!",
        "docs": "/docs",
        "health": "/health",
    }
