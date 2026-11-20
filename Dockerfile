# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# ---- base: dependencies only -------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --system --uid 10001 --user-group --no-create-home appuser \
 && mkdir -p /data \
 && chown appuser:appuser /data
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- test: runs the whole unittest suite on the real runtime (docker build --target test) ----
FROM base AS test
COPY . .
USER appuser
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]

# ---- runtime: application code only (the default target; must stay LAST) --------
FROM base AS runtime
COPY main.py ./
COPY models ./models
COPY routers ./routers
COPY services ./services
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
