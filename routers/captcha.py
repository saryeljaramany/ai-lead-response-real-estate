"""
routers/captcha.py

HTTP routing layer for Cloudflare Turnstile CAPTCHA verification.

This file contains zero business logic. Its sole responsibility is to:
    1. Accept HTTP requests and validate inputs at the routing layer.
    2. Delegate all verification logic to services/captcha_service.py.
    3. Map service return values and exceptions to appropriate HTTP responses.

Endpoint summary:
    POST /captcha/verify  -- Verify a Cloudflare Turnstile token server-side.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from services.captcha_service import CaptchaError, verify_token


router = APIRouter(prefix="/captcha", tags=["CAPTCHA"])


# ===========================================================================
# REQUEST & RESPONSE MODELS
# ===========================================================================

class CaptchaVerifyRequest(BaseModel):
    """HTTP request payload for POST /captcha/verify."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token": "0.XXXX.DUMMY.TOKEN.XXXX",
                "remote_ip": "203.0.113.9",
            }
        }
    )

    token: str | None = None
    remote_ip: str | None = None


class CaptchaVerifyResponse(BaseModel):
    """HTTP response body returned by POST /captcha/verify."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "verified": True,
                "reason": "ok",
                "error_codes": [],
            }
        }
    )

    verified: bool
    reason: str
    error_codes: list[str]


# ===========================================================================
# ENDPOINT -- Verify Turnstile Token
# ===========================================================================

@router.post(
    "/verify",
    response_model=CaptchaVerifyResponse,
    summary="Verify a Cloudflare Turnstile CAPTCHA token",
    responses={
        200: {
            "description": (
                "Token evaluation complete. Inspect the ``verified`` boolean "
                "and ``reason`` field to determine whether verification succeeded."
            ),
        },
        422: {
            "description": (
                "Request body validation failed (e.g. non-string token or IP). "
                "Correct the request structure before retrying."
            ),
        },
        500: {
            "description": (
                "Configuration error (e.g. missing or rejected TURNSTILE_SECRET_KEY) "
                "or unexpected internal error. Retrying will not help until configuration is fixed."
            ),
        },
        502: {
            "description": (
                "The CAPTCHA provider is temporarily unavailable or a network timeout "
                "occurred. This response is retryable."
            ),
        },
    },
)
def verify_captcha_endpoint(payload: CaptchaVerifyRequest) -> CaptchaVerifyResponse:
    """Verify a Cloudflare Turnstile token server-side.

    Delegates verification to ``verify_token`` in ``services/captcha_service.py``.
    Declared as a plain ``def`` (not ``async def``) because the underlying
    HTTP call to the CAPTCHA provider is blocking; FastAPI runs synchronous
    handlers in a thread pool so the event loop is never blocked.

    **Verification Outcomes**

    - ``200`` with ``verified=True`` -- token is valid and verified.
    - ``200`` with ``verified=False`` -- token is missing, invalid, expired,
      rejected, or hostname mismatched.
    - ``502`` -- provider or network temporarily unavailable (retryable).
    - ``500`` -- provider configuration error or unexpected server error.

    **Privacy**

    Neither the token, the secret key, the client remote IP, nor the provider's
    internal body text is ever echoed in any response body or error detail.
    """
    try:
        result = verify_token(payload.token, payload.remote_ip)
        return CaptchaVerifyResponse(**result)
    except CaptchaError as exc:
        status_code = 502 if exc.category == "transient" else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while verifying the CAPTCHA.",
        ) from exc
