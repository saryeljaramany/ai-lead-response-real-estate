"""
services/gmail_service.py

Gmail OAuth 2.0 integration layer for the AI-powered lead management system.

This module handles the complete Gmail lifecycle:
    1. OAuth 2.0 authorization URL generation
    2. Authorization code exchange for tokens
    3. Token persistence (token.json in the project root)
    4. Automatic token refresh when credentials expire
    5. Email dispatch via the Gmail API

All environment variables are read inside functions at call time - never at
module level - to guarantee that load_dotenv() in main.py has already run
before any value is accessed.

Dry-run mode
------------
Set GMAIL_DRY_RUN to any value other than "false", "0", "no", or "off"
(case-insensitive, stripped) and send_email() will return immediately without
reading credentials, loading token.json, or touching the network. This is the
default-safe mode: an unset, empty, or misspelled variable is treated as True
so that accidental sends cannot occur unless the flag is explicitly disabled.

Error categories (EmailSendError.category)
------------------------------------------
    "recipient"  - Gmail rejected the message or recipient address; do not retry.
    "config"     - Authorization, scope, sender or server setup problem; do not retry.
    "transient"  - Gmail or the network failed transiently; a retry is reasonable.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import base64
import json
import os
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import google.auth.exceptions
import google.auth.transport.requests
import google.oauth2.credentials
import google_auth_httplib2
import httplib2
from google.auth.exceptions import RefreshError, TransportError
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from models import EmailPayload


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The only scope needed: permission to send email on behalf of the user.
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

# Absolute path to token.json in the project root (one level above services/).
TOKEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "token.json",
)

# Hard timeout (seconds) for every Gmail HTTP call.
_REQUEST_TIMEOUT_SECONDS = 10


# ---------------------------------------------------------------------------
# Typed error
# ---------------------------------------------------------------------------

class EmailSendError(RuntimeError):
    """
    Raised by send_email() when the Gmail API call cannot be completed.

    Attributes:
        category: One of "recipient", "config", or "transient".  The router
                  uses this to choose the appropriate HTTP status code and to
                  decide whether the caller should retry.
    """

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """
    Read an environment variable at call time, strip it, and return it.

    This function exists to centralise the "read-and-validate" pattern so
    that every caller gets the same clear error message when a variable is
    missing.

    Parameters:
        name: The environment variable name to look up.

    Returns:
        The stripped, non-empty value of the variable.

    Raises:
        EmailSendError(category="config"): If the variable is unset, empty,
            or whitespace-only.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise EmailSendError(
            f"{name} is missing or empty. Set it in .env and restart the server.",
            category="config",
        )
    return value


def _is_dry_run() -> bool:
    """
    Return True unless GMAIL_DRY_RUN is explicitly set to a falsy value.

    Reads GMAIL_DRY_RUN at call time (never cached).  Returns False ONLY when
    the stripped, lower-cased value is one of: "false", "0", "no", "off".
    Any other value - including unset, empty, or a typo - returns True, so
    that accidental live sends cannot occur without an explicit opt-in.

    Returns:
        bool: True if dry-run mode is active, False for live mode.
    """
    raw = os.getenv("GMAIL_DRY_RUN", "").strip().lower()
    return raw not in {"false", "0", "no", "off"}


def _classify_http_error(exc: HttpError) -> EmailSendError:
    """
    Convert a googleapiclient HttpError into a typed EmailSendError.

    Maps HTTP status codes (and, for 403, specific reason strings) to the
    appropriate error category.  Error messages never contain the recipient
    address, subject, body, or any text from Gmail's response body.

    Parameters:
        exc: The HttpError raised by the Gmail API client.

    Returns:
        An EmailSendError with an appropriate category and message.
    """
    status = int(exc.resp.status)

    # Try to extract reason strings from the JSON error body.
    reasons: list[str] = []
    try:
        body = json.loads(exc.content)
        for entry in body.get("error", {}).get("errors", []):
            reason = entry.get("reason", "")
            if reason:
                reasons.append(reason)
    except Exception:  # noqa: BLE001 - tolerate any parse failure
        pass

    if status == 400:
        return EmailSendError(
            "Gmail rejected the message or its recipient address (400).",
            category="recipient",
        )
    if status == 401:
        return EmailSendError(
            "Gmail authorization failed (401). Re-run GET /email/auth.",
            category="config",
        )
    if status == 403:
        rate_limit_reasons = {"rateLimitExceeded", "userRateLimitExceeded"}
        if rate_limit_reasons.intersection(reasons):
            return EmailSendError(
                "Gmail rate limit reached (403). Retry later.",
                category="transient",
            )
        return EmailSendError(
            "Gmail denied access (403). Check the OAuth scope, "
            "the sender account and its sending quota.",
            category="config",
        )
    if status == 404:
        return EmailSendError(
            "Gmail endpoint or sender account not found (404).",
            category="config",
        )
    if status == 429:
        return EmailSendError(
            "Gmail rate limit reached (429). Retry later.",
            category="transient",
        )
    if status >= 500:
        return EmailSendError(
            "Gmail is temporarily unavailable. Retry later.",
            category="transient",
        )
    return EmailSendError(
        f"Gmail rejected the request (HTTP {status}).",
        category="config",
    )


def _build_flow() -> Flow:
    """
    Construct a new Google OAuth 2.0 Flow from environment variables.

    The client configuration is assembled manually (no client_secrets.json
    file required) using GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and
    GMAIL_REDIRECT_URI from the environment.

    PKCE is explicitly disabled (autogenerate_code_verifier=False) because
    this is a confidential web client that authenticates with a client secret,
    and because the authorization and token-exchange requests arrive in
    separate HTTP requests - a per-Flow PKCE verifier would be lost between
    them, causing Google to reject the exchange with "invalid_grant: Missing
    code verifier".

    Both get_auth_url() and exchange_code() call this helper independently
    so that each receives a fresh, identically configured Flow.

    Returns:
        A fully configured google_auth_oauthlib.flow.Flow instance.

    Raises:
        RuntimeError: If any required environment variable is missing.
    """
    client_id = _require_env("GMAIL_CLIENT_ID")
    client_secret = _require_env("GMAIL_CLIENT_SECRET")
    redirect_uri = _require_env("GMAIL_REDIRECT_URI")

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }

    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        autogenerate_code_verifier=False,
    )


def _to_naive_utc(dt: datetime | None) -> datetime | None:
    """
    Normalise a datetime to a naive UTC datetime, or pass through None.

    google-auth stores credentials.expiry as a naive datetime assumed to be
    in UTC.  This helper ensures that any timezone-aware datetime coming from
    serialisation round-trips is converted to the same naive-UTC form before
    being handed back to Credentials.

    Parameters:
        dt: A datetime (naive or aware) or None.

    Returns:
        A naive UTC datetime, or None if the input was None.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _save_credentials(credentials: google.oauth2.credentials.Credentials) -> None:
    """
    Persist OAuth 2.0 credentials to token.json in the project root.

    The file is written with UTF-8 encoding and pretty-printed (indent=2)
    for easy manual inspection during development.

    Expiry format: google-auth keeps credentials.expiry as a naive datetime
    in UTC. We write it as an ISO 8601 string with a trailing "Z" - the same
    convention used by google-auth's own Credentials.to_json().

    Parameters:
        credentials: A google.oauth2.credentials.Credentials instance whose
                     token and refresh_token should be persisted.

    Raises:
        OSError: If the file cannot be written.
    """
    expiry = _to_naive_utc(credentials.expiry)
    expiry_str = expiry.isoformat() + "Z" if expiry is not None else None

    data = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes) if credentials.scopes else [],
        "expiry": expiry_str,
    }

    with open(TOKEN_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_credentials() -> google.oauth2.credentials.Credentials:
    """
    Load OAuth 2.0 credentials from token.json, refreshing if expired.

    Workflow:
        1. Read and parse token.json.
        2. Reconstruct the Credentials object with a naive-UTC expiry.
        3. If the credentials are expired and a refresh token is available,
           attempt a silent refresh and persist the updated tokens.
        4. Return the valid Credentials.

    Returns:
        A google.oauth2.credentials.Credentials instance ready for use
        with the Gmail API.

    Raises:
        EmailSendError(category="config"): If token.json does not exist,
            contains invalid JSON, or the refresh attempt fails (e.g. the
            user revoked access).
    """
    # --- 1. Existence check ---
    if not os.path.exists(TOKEN_PATH):
        raise EmailSendError(
            "token.json not found. Complete the OAuth flow first by visiting "
            "GET /email/auth",
            category="config",
        )

    # --- 2. Parse ---
    try:
        with open(TOKEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError) as exc:
        raise EmailSendError(
            "token.json contains invalid JSON. Delete the file and re-run "
            "GET /email/auth to generate a fresh token.",
            category="config",
        ) from exc

    # --- 3. Expiry reconstruction ---
    raw_expiry = data.get("expiry")
    expiry: datetime | None = None
    if raw_expiry and isinstance(raw_expiry, str):
        # Replace trailing "Z" with "+00:00" for Python 3.10 compatibility
        # (fromisoformat does not accept "Z" until Python 3.11).
        expiry = _to_naive_utc(
            datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
        )

    # --- 4. Build Credentials ---
    credentials = google.oauth2.credentials.Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
        expiry=expiry,
    )

    # --- 5. Refresh if expired ---
    if credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(google.auth.transport.requests.Request())
        except google.auth.exceptions.RefreshError as exc:
            raise EmailSendError(
                "Gmail authorization expired or was revoked. Re-run "
                "GET /email/auth.",
                category="config",
            ) from exc
        _save_credentials(credentials)

    return credentials


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def get_auth_url() -> str:
    """
    Generate the Google OAuth 2.0 authorization URL.

    The user (or developer) should visit this URL in a browser, grant the
    requested Gmail send permission, and be redirected back to the
    application's callback endpoint with an authorization code.

    The authorization is requested with ``access_type="offline"`` and
    ``prompt="consent"`` so that Google always issues a refresh token -
    even if the user has previously granted access.

    Returns:
        The full authorization URL as a string.

    Raises:
        RuntimeError: If any required environment variable
                      (GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET,
                      GMAIL_REDIRECT_URI) is missing.
    """
    flow = _build_flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    return auth_url


def exchange_code(code: str) -> None:
    """
    Exchange an authorization code for OAuth 2.0 tokens and persist them.

    This function is called by the OAuth callback endpoint after the user
    grants permission. It exchanges the code for an access token and a
    refresh token, then writes them to token.json so that subsequent
    send_email() calls can authenticate without user interaction.

    Parameters:
        code: The authorization code returned by Google in the callback
              query string.

    Raises:
        RuntimeError: If the code exchange fails (e.g. the code is expired
                      or has already been used) or if a required environment
                      variable is missing.
    """
    try:
        flow = _build_flow()
        flow.fetch_token(code=code)
        _save_credentials(flow.credentials)
    except RuntimeError:
        # Re-raise our own RuntimeErrors (from _require_env / _build_flow)
        # unchanged - they already carry user-actionable text.
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to exchange authorization code: {exc}"
        ) from exc


def send_email(payload: EmailPayload) -> dict:
    """
    Send an email via the Gmail API using a previously authorized account.

    Dry-run mode
    ------------
    If GMAIL_DRY_RUN is not explicitly set to "false", "0", "no", or "off"
    (case-insensitive), the function returns immediately without reading any
    other environment variable, loading token.json, or touching the network.
    The returned dict includes ``"dry_run": True`` and ``"message_id": None``.
    Set ``GMAIL_DRY_RUN=false`` to enable live sending.

    Live mode workflow
    ------------------
    1. Validate that GMAIL_SENDER_EMAIL is configured.
    2. Load (and, if necessary, refresh) OAuth credentials from token.json.
    3. Build an AuthorizedHttp transport with a hard timeout of
       _REQUEST_TIMEOUT_SECONDS seconds for every Gmail HTTP call.
    4. Build the email as a MIMEMultipart message.
    5. Base64url-encode the message bytes and dispatch via the Gmail API.
    6. Return a confirmation dict with the recipient, subject, Gmail message
       ID, and ``"dry_run": False``.

    Parameters:
        payload: An EmailPayload instance containing to_email, to_name,
                 subject, body, lead_score, and an optional calendly_link.

    Returns:
        A dict with keys: "message", "to", "subject", "message_id",
        "dry_run".

    Raises:
        EmailSendError(category="recipient"): Gmail rejected the message or
            its recipient address (HTTP 400).  Do not retry.
        EmailSendError(category="config"): Authorization, scope, sender or
            server setup problem (HTTP 401, 403 non-rate-limit, 404, or an
            unexpected exception type).  Do not retry; fix the configuration.
        EmailSendError(category="transient"): Gmail or the network failed
            transiently (HTTP 429, 5xx, network timeout, or OS error).  A
            retry after a back-off delay is reasonable.
    """
    # Normalise the subject to a single line (RFC 2822 forbids bare newlines
    # in structured header fields).
    subject = " ".join(payload.subject.split())

    # --- Dry-run short-circuit ---
    if _is_dry_run():
        return {
            "message": (
                "DRY RUN: email prepared; nothing was sent. "
                "Set GMAIL_DRY_RUN=false to send real email."
            ),
            "to": payload.to_email,
            "subject": subject,
            "message_id": None,
            "dry_run": True,
        }

    # --- Live send ---
    try:
        # 1. Validate sender - before touching token.json.
        sender = _require_env("GMAIL_SENDER_EMAIL")

        # 2. Load credentials (auto-refreshes if expired).
        credentials = _load_credentials()

        # 3. Build the Gmail API service with an explicit HTTP timeout.
        http = google_auth_httplib2.AuthorizedHttp(
            credentials,
            http=httplib2.Http(timeout=_REQUEST_TIMEOUT_SECONDS),
        )
        service = build("gmail", "v1", http=http)

        # 4. Compose the MIME message.
        message = MIMEMultipart()
        message["To"] = payload.to_email
        message["From"] = sender
        message["Subject"] = subject
        message.attach(MIMEText(payload.body, "plain", "utf-8"))

        # 5. Encode and send.
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        response = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": raw})
            .execute()
        )

        # 6. Return confirmation.
        return {
            "message": "Email sent successfully",
            "to": payload.to_email,
            "subject": subject,
            "message_id": response.get("id"),
            "dry_run": False,
        }

    except EmailSendError:
        raise
    except HttpError as exc:
        raise _classify_http_error(exc) from exc
    except RefreshError as exc:
        raise EmailSendError(
            "Gmail authorization expired or was revoked. Re-run GET /email/auth.",
            category="config",
        ) from exc
    except (TransportError, httplib2.HttpLib2Error, OSError) as exc:
        raise EmailSendError(
            "Could not reach Gmail (network error or timeout).",
            category="transient",
        ) from exc
    except Exception as exc:
        raise EmailSendError(
            f"Unexpected error while calling Gmail: {type(exc).__name__}.",
            category="config",
        ) from exc
