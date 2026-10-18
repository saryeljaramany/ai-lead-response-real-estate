# =============================================================================
# Imports
# =============================================================================
from fastapi import FastAPI
from dotenv import load_dotenv
import os

from routers import leads, ai

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
)

# =============================================================================
# REGISTER ROUTERS
# =============================================================================
app.include_router(leads.router)  # handles /leads routes
app.include_router(ai.router)     # handles /ai routes

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
