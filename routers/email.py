"""
routers/email.py

HTTP routing layer for the Gmail OAuth 2.0 integration and email dispatch.

This file contains zero business logic. Its sole responsibility is to:
    1. Accept HTTP requests and validate inputs at the routing layer.
    2. Delegate all Gmail logic to services/gmail_service.py.
    3. Map service return values and exceptions to appropriate HTTP responses.

OAuth flow overview:
    GET  /email/auth          → Step 1: generate a Google consent URL
    GET  /email/auth/callback → Step 2: exchange the code Google returns
    POST /email/send          → Step 3: send an email with stored credentials
"""

from fastapi import APIRouter, HTTPException, Query

from models import EmailPayload
from services.gmail_service import exchange_code, get_auth_url, send_email

router = APIRouter(
    prefix="/email",
    tags=["Email"],
)


# ===========================================================================
# ENDPOINT 1 — Generate Authorization URL
# ===========================================================================

@router.get("/auth", summary="Start Gmail OAuth 2.0 authorization")
async def auth() -> dict:
    """
    Generate a Google OAuth 2.0 authorization URL and return it to the caller.

    This is Step 1 of the Gmail authorization flow. The developer (or an
    automated process) must open the returned URL in a browser, sign in with
    the Google account that owns the Gmail inbox, and approve the requested
    ``gmail.send`` permission.

    After approval, Google redirects the browser to the configured
    GMAIL_REDIRECT_URI (typically GET /email/auth/callback) with an
    authorization code in the query string. That code must be exchanged
    via GET /email/auth/callback to obtain and persist OAuth tokens.

    **Receives:**
        Nothing — no request body or query parameters.

    **Returns:**
        A JSON object with:
        - ``message``: instructions for the developer.
        - ``auth_url``: the full Google consent URL to open in a browser.

    **Errors:**
        - ``500 Internal Server Error`` — a required environment variable
          (GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, or GMAIL_REDIRECT_URI)
          is missing, or an unexpected error occurred in the service layer.
    """
    try:
        url = get_auth_url()
        return {
            "message": (
                "Open the auth_url in a browser, sign in with the Gmail account "
                "you want to send from, and approve the requested access. "
                "You will be redirected to GET /email/auth/callback automatically."
            ),
            "auth_url": url,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while generating the authorization URL.",
        ) from exc


# ===========================================================================
# ENDPOINT 2 — OAuth Callback (Code Exchange)
# ===========================================================================

@router.get("/auth/callback", summary="Complete Gmail OAuth 2.0 authorization")
async def auth_callback(
    code: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> dict:
    """
    Receive Google's OAuth 2.0 callback and exchange the authorization code
    for access and refresh tokens.

    This is Step 2 of the Gmail authorization flow. Google redirects here
    after the user either approves or denies consent on the authorization
    page.

    Both ``code`` and ``error`` are declared optional because Google's
    redirect behaviour differs by outcome:
    - **Approved**: Google redirects with ``?code=<auth_code>`` and no error.
    - **Denied**: Google redirects with ``?error=access_denied`` and no code.
    If ``code`` were declared as a required ``str``, FastAPI would return a
    ``422`` before this handler runs whenever consent is denied, and the
    user-friendly denial message could never be produced.

    Validation is performed **before** the ``try`` block so that
    ``HTTPException`` (a non-Exception) is never caught by the broad
    ``except Exception`` clause.

    **Receives:**
        Query parameters (both optional, populated by Google's redirect):
        - ``code``  — authorization code to exchange for tokens.
        - ``error`` — denial reason string (e.g. ``access_denied``).

    **Returns:**
        On success:
        ``{"message": "Authorization successful. You can now send emails via POST /email/send"}``

    **Errors:**
        - ``400 Bad Request`` — the user denied consent (``error`` is present)
          or the request arrived without a ``code`` and without an ``error``
          (e.g. direct navigation to this URL).
        - ``500 Internal Server Error`` — the code exchange failed (expired,
          already used, or environment misconfiguration), or an unexpected
          error occurred.
    """
    # --- Validate before try so HTTPExceptions are never swallowed ---
    if error:
        raise HTTPException(
            status_code=400,
            detail=f"Authorization denied by user: {error}",
        )
    if not code:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing 'code' query parameter. "
                "Start again at GET /email/auth."
            ),
        )

    try:
        exchange_code(code)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while completing authorization.",
        ) from exc

    return {
        "message": (
            "Authorization successful. "
            "You can now send emails via POST /email/send"
        )
    }


# ===========================================================================
# ENDPOINT 3 — Send Email
# ===========================================================================

@router.post("/send", summary="Send a follow-up email to a lead via Gmail")
async def send_email_endpoint(payload: EmailPayload) -> dict:
    """
    Send a personalised follow-up email to a lead using the authorized Gmail
    account and the credentials stored in token.json.

    This is Step 3 of the Gmail integration. Credentials must already exist
    (i.e. the OAuth flow must have been completed via GET /email/auth and
    GET /email/auth/callback). If the stored access token is expired, the
    service layer will silently refresh it using the refresh token before
    dispatching the email.

    ``RuntimeError`` messages from the service layer are forwarded verbatim as
    HTTP 500 detail strings because they contain user-actionable instructions
    (e.g. "token.json not found. Complete the OAuth flow first..."). Any other
    unexpected exception is wrapped in a generic message to avoid leaking
    internal details.

    **Receives:**
        An ``EmailPayload`` JSON request body:
        - ``to_email`` (required) — recipient email address.
        - ``to_name`` (required) — recipient display name.
        - ``subject`` (required) — email subject line.
        - ``body`` (required) — full email body text.
        - ``lead_score`` (required) — AI quality score (0–100); used for
          logging, not included in the email.
        - ``calendly_link`` (optional) — booking URL; include in the body
          when present.

    **Returns:**
        A dict with:
        - ``message``: ``"Email sent successfully"``
        - ``to``: recipient email address
        - ``subject``: email subject line
        - ``message_id``: Gmail API message ID for audit/tracking

    **Errors:**
        - ``422 Unprocessable Entity`` — ``EmailPayload`` validation failed
          (FastAPI handles this automatically before the handler runs).
        - ``500 Internal Server Error`` — token.json is missing or corrupt,
          the refresh token was revoked, GMAIL_SENDER_EMAIL is not set, or
          the Gmail API call failed.
    """
    try:
        result = send_email(payload)
        return result
    except RuntimeError as exc:
        # RuntimeErrors from our service layer carry user-actionable text;
        # forward them verbatim so the caller knows exactly what to fix.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while sending the email. Please try again.",
        ) from exc
