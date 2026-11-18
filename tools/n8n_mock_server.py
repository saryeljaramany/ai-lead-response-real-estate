"""n8n mock backend and Slack server.

Test double providing deterministic responses for every branch of the n8n
lead processing pipeline workflow.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time
import uuid


def _is_valid_uuid(val):
    if not isinstance(val, str):
        return False
    try:
        uuid.UUID(val)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _make_step(
    name,
    status,
    reason,
    retryable=False,
    external_id=None,
    flag_recorded=None,
    duration_ms=0,
    error=None,
):
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "retryable": retryable,
        "external_id": external_id,
        "flag_recorded": flag_recorded,
        "duration_ms": duration_ms,
        "error": error,
    }


def _build_completed_hot(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step(
            "email",
            "succeeded",
            "sent",
            external_id="m1",
            flag_recorded=True,
            duration_ms=15,
        ),
        _make_step(
            "sms",
            "succeeded",
            "sent",
            external_id="SM1",
            flag_recorded=True,
            duration_ms=12,
        ),
        _make_step(
            "crm",
            "succeeded",
            "synced",
            external_id="512",
            flag_recorded=True,
            duration_ms=20,
        ),
    ]
    return {
        "lead_id": lead_id,
        "status": "completed",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "hot",
        "score": 87,
        "steps": steps,
        "retry": {"recommended": False, "steps": []},
        "repair": [],
        "failed_step": None,
        "retryable": False,
    }


def _build_completed_warm(lead_id):
    rep = _build_completed_hot(lead_id)
    rep["category"] = "warm"
    rep["score"] = 55
    return rep


def _build_completed_cold(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step("email", "skipped", "cold_lead", duration_ms=0),
        _make_step("sms", "skipped", "cold_lead", duration_ms=0),
        _make_step("crm", "skipped", "cold_lead", duration_ms=0),
    ]
    return {
        "lead_id": lead_id,
        "status": "completed",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "cold",
        "score": 20,
        "steps": steps,
        "retry": {"recommended": False, "steps": []},
        "repair": [],
        "failed_step": None,
        "retryable": False,
    }


def _build_completed_dry_run(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step("email", "skipped", "dry_run", duration_ms=0),
        _make_step("sms", "skipped", "dry_run", duration_ms=0),
        _make_step("crm", "skipped", "dry_run", duration_ms=0),
    ]
    return {
        "lead_id": lead_id,
        "status": "completed",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "hot",
        "score": 87,
        "steps": steps,
        "retry": {"recommended": False, "steps": []},
        "repair": [],
        "failed_step": None,
        "retryable": False,
    }


def _build_score_failed_transient(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step(
            "score",
            "failed",
            "transient",
            retryable=True,
            duration_ms=10,
            error={
                "category": "transient",
                "message": (
                    "Scoring service is temporarily unavailable. Retry later."
                ),
            },
        ),
        _make_step(
            "store",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
        _make_step(
            "email",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
        _make_step(
            "sms",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
        _make_step(
            "crm",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
    ]
    return {
        "lead_id": lead_id,
        "status": "failed",
        "persisted": False,
        "created": None,
        "idempotency_key_generated": False,
        "category": None,
        "score": None,
        "steps": steps,
        "retry": {"recommended": True, "steps": ["score"]},
        "repair": [],
        "failed_step": "score",
        "retryable": True,
    }


def _build_score_failed_config(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step(
            "score",
            "failed",
            "config",
            retryable=False,
            duration_ms=10,
            error={
                "category": "config",
                "message": (
                    "Scoring provider configuration is missing or invalid."
                ),
            },
        ),
        _make_step(
            "store",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
        _make_step(
            "email",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
        _make_step(
            "sms",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
        _make_step(
            "crm",
            "skipped",
            "upstream_failed",
            retryable=False,
            duration_ms=0,
        ),
    ]
    return {
        "lead_id": lead_id,
        "status": "failed",
        "persisted": False,
        "created": None,
        "idempotency_key_generated": False,
        "category": None,
        "score": None,
        "steps": steps,
        "retry": {"recommended": False, "steps": []},
        "repair": [],
        "failed_step": "score",
        "retryable": False,
    }


def _build_partial_retry(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step(
            "email",
            "succeeded",
            "sent",
            external_id="m1",
            flag_recorded=True,
            duration_ms=15,
        ),
        _make_step(
            "sms",
            "failed",
            "transient",
            retryable=True,
            external_id=None,
            flag_recorded=False,
            duration_ms=12,
            error={
                "category": "transient",
                "message": (
                    "Twilio is rate limiting or temporarily unavailable. "
                    "Retry later."
                ),
            },
        ),
        _make_step(
            "crm",
            "succeeded",
            "synced",
            external_id="512",
            flag_recorded=True,
            duration_ms=20,
        ),
    ]
    return {
        "lead_id": lead_id,
        "status": "partial",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "hot",
        "score": 87,
        "steps": steps,
        "retry": {"recommended": True, "steps": ["sms"]},
        "repair": [],
        "failed_step": "sms",
        "retryable": True,
    }


def _build_repair_then_ok(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step(
            "email",
            "failed",
            "flag_update_failed",
            retryable=False,
            external_id="m1",
            flag_recorded=False,
            duration_ms=15,
            error={
                "category": "storage",
                "message": "Failed to update email_sent flag in database.",
            },
        ),
        _make_step(
            "sms",
            "succeeded",
            "sent",
            external_id="SM1",
            flag_recorded=True,
            duration_ms=12,
        ),
        _make_step(
            "crm",
            "succeeded",
            "synced",
            external_id="512",
            flag_recorded=True,
            duration_ms=20,
        ),
    ]
    action_str = "PATCH /leads/" + str(lead_id) + "/flags"
    return {
        "lead_id": lead_id,
        "status": "partial",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "hot",
        "score": 87,
        "steps": steps,
        "retry": {"recommended": False, "steps": []},
        "repair": [{"action": action_str, "body": {"email_sent": True}}],
        "failed_step": "email",
        "retryable": False,
    }


def _build_repair_and_retry(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step(
            "email",
            "failed",
            "flag_update_failed",
            retryable=False,
            external_id="m1",
            flag_recorded=False,
            duration_ms=15,
            error={
                "category": "storage",
                "message": "Failed to update email_sent flag in database.",
            },
        ),
        _make_step(
            "sms",
            "failed",
            "transient",
            retryable=True,
            external_id=None,
            flag_recorded=False,
            duration_ms=12,
            error={
                "category": "transient",
                "message": (
                    "Twilio is rate limiting or temporarily unavailable. "
                    "Retry later."
                ),
            },
        ),
        _make_step(
            "crm",
            "succeeded",
            "synced",
            external_id="512",
            flag_recorded=True,
            duration_ms=20,
        ),
    ]
    action_str = "PATCH /leads/" + str(lead_id) + "/flags"
    return {
        "lead_id": lead_id,
        "status": "partial",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "hot",
        "score": 87,
        "steps": steps,
        "retry": {"recommended": True, "steps": ["sms"]},
        "repair": [{"action": action_str, "body": {"email_sent": True}}],
        "failed_step": "email",
        "retryable": True,
    }


def _build_partial_not_retryable(lead_id):
    steps = [
        _make_step("intake", "succeeded", "normalized", duration_ms=2),
        _make_step("score", "succeeded", "scored", duration_ms=10),
        _make_step("store", "succeeded", "created", duration_ms=5),
        _make_step(
            "email",
            "succeeded",
            "sent",
            external_id="m1",
            flag_recorded=True,
            duration_ms=15,
        ),
        _make_step(
            "sms",
            "succeeded",
            "sent",
            external_id="SM1",
            flag_recorded=True,
            duration_ms=12,
        ),
        _make_step(
            "crm",
            "failed",
            "config",
            retryable=False,
            external_id=None,
            flag_recorded=False,
            duration_ms=20,
            error={
                "category": "config",
                "message": (
                    "HubSpot configuration error: token is invalid or missing."
                ),
            },
        ),
    ]
    return {
        "lead_id": lead_id,
        "status": "partial",
        "persisted": True,
        "created": True,
        "idempotency_key_generated": False,
        "category": "hot",
        "score": 87,
        "steps": steps,
        "retry": {"recommended": False, "steps": []},
        "repair": [],
        "failed_step": "crm",
        "retryable": False,
    }


class ServerState:
    """Thread-safe state container for mock server events and call counters."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.events = []
        self.calls_by_id = {}
        self.calls_by_scenario = {}
        self.scenario_by_id = {}

    def reset(self):
        with self.lock:
            self.seq = 0
            self.events = []
            self.calls_by_id = {}
            self.calls_by_scenario = {}
            self.scenario_by_id = {}

    def record_process(self, lead_id, scenario):
        with self.lock:
            attempt = self.calls_by_id.get(lead_id, 0) + 1
            self.calls_by_id[lead_id] = attempt
            cur_scenario_count = self.calls_by_scenario.get(scenario, 0) + 1
            self.calls_by_scenario[scenario] = cur_scenario_count
            self.scenario_by_id[lead_id] = scenario
            self.seq += 1
            event = {
                "seq": self.seq,
                "kind": "process",
                "id": lead_id,
                "scenario": scenario,
                "attempt": attempt,
                "payload": None,
            }
            self.events.append(event)
            return attempt

    def record_patch(self, lead_id, payload):
        with self.lock:
            scenario = self.scenario_by_id.get(lead_id)
            self.seq += 1
            event = {
                "seq": self.seq,
                "kind": "patch",
                "id": lead_id,
                "scenario": scenario,
                "attempt": None,
                "payload": payload,
            }
            self.events.append(event)
            return scenario

    def record_slack(self, kind, payload):
        with self.lock:
            self.seq += 1
            event = {
                "seq": self.seq,
                "kind": kind,
                "id": None,
                "scenario": None,
                "attempt": None,
                "payload": payload,
            }
            self.events.append(event)

    def get_events(self):
        with self.lock:
            return [dict(e) for e in self.events]


_KNOWN_FIXED_SCENARIOS = {
    "completed-hot",
    "completed-warm",
    "completed-cold",
    "completed-dry-run",
    "transient-then-ok",
    "partial-retry-then-ok",
    "always-502",
    "repair-then-ok",
    "repair-and-retry",
    "repair-fail",
    "terminal-config",
    "partial-not-retryable",
    "validation-422",
    "drop-connection",
}


class MockHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock backend and Slack endpoints."""

    def log_message(self, format, *args):
        # Silence default per-request log messages
        pass

    @property
    def state(self):
        return self.server.state

    def _read_body_bytes(self):
        content_length = self.headers.get("Content-Length")
        if not content_length:
            return b""
        try:
            length = int(content_length)
            return self.rfile.read(length)
        except (ValueError, TypeError):
            return b""

    def _read_json_body(self):
        raw = self._read_body_bytes()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _get_clean_path(self):
        raw_path = self.path.split("?")[0]
        cleaned = raw_path.rstrip("/")
        if not cleaned:
            return "/"
        return cleaned

    def do_GET(self):
        path = self._get_clean_path()
        if path == "/health":
            self._send_json(200, {"status": "ok"})
        elif path == "/_log":
            self._send_json(200, self.state.get_events())
        else:
            self._send_json(404, {"detail": "not found"})

    def do_POST(self):
        path = self._get_clean_path()
        if path == "/leads/process":
            self._handle_leads_process()
        elif path == "/slack/leads":
            body = self._read_json_body()
            self.state.record_slack("slack_leads", body)
            self._send_text(200, "ok")
        elif path == "/slack/alerts":
            body = self._read_json_body()
            self.state.record_slack("slack_alerts", body)
            self._send_text(200, "ok")
        elif path == "/_reset":
            self.state.reset()
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"detail": "not found"})

    def do_PATCH(self):
        path = self._get_clean_path()
        match = re.match(r"^/leads/([^/]+)/flags$", path)
        if not match:
            self._send_json(404, {"detail": "not found"})
            return

        lead_id = match.group(1)
        body = self._read_json_body()
        scenario = self.state.record_patch(lead_id, body)
        if scenario == "repair-fail":
            self._send_json(500, {"detail": "boom"})
        else:
            self._send_json(200, {"id": lead_id, "ok": True})

    def _handle_leads_process(self):
        body = self._read_json_body()
        if body is None or not isinstance(body, dict):
            self._send_json(422, {"detail": "invalid json body"})
            return

        req_id = body.get("id")
        lead = body.get("lead")

        errors = []
        if req_id is None:
            errors.append(
                {"loc": ["body", "id"], "msg": "field required", "input": None}
            )
        elif not _is_valid_uuid(req_id):
            errors.append(
                {
                    "loc": ["body", "id"],
                    "msg": "value is not a valid uuid",
                    "input": req_id,
                }
            )

        if lead is None or not isinstance(lead, dict):
            errors.append(
                {"loc": ["body", "lead"], "msg": "field required", "input": lead}
            )
        else:
            for field in ("name", "email", "phone", "message"):
                val = lead.get(field)
                if val is None or not isinstance(val, str):
                    errors.append(
                        {
                            "loc": ["body", "lead", field],
                            "msg": "field required",
                            "input": val,
                        }
                    )

        if errors:
            self._send_json(422, {"detail": errors})
            return

        # Extract scenario from lead message
        message = lead.get("message", "")
        marker_match = re.search(r"\[scenario:([a-zA-Z0-9_-]+)\]", message)
        if marker_match:
            scenario = marker_match.group(1).strip()
        else:
            scenario = "completed-hot"

        is_slow = False
        slow_seconds = 0
        slow_match = re.match(r"^slow-(\d+)$", scenario)
        if slow_match:
            is_slow = True
            slow_seconds = int(slow_match.group(1))
            if slow_seconds > 600:
                self._send_json(400, {"detail": "unknown scenario"})
                return
        elif scenario not in _KNOWN_FIXED_SCENARIOS:
            self._send_json(400, {"detail": "unknown scenario"})
            return

        # Record attempt for this id and scenario
        attempt = self.state.record_process(req_id, scenario)

        # Handle drop-connection immediately after recording
        if scenario == "drop-connection":
            self.close_connection = True
            try:
                self.request.close()
            except OSError:
                pass
            return

        # Handle slow-<n>
        if is_slow:
            time.sleep(slow_seconds)
            self._send_json(200, _build_completed_hot(req_id))
            return

        # Handle fixed scenarios
        if scenario == "completed-hot":
            self._send_json(200, _build_completed_hot(req_id))
        elif scenario == "completed-warm":
            self._send_json(200, _build_completed_warm(req_id))
        elif scenario == "completed-cold":
            self._send_json(200, _build_completed_cold(req_id))
        elif scenario == "completed-dry-run":
            self._send_json(200, _build_completed_dry_run(req_id))
        elif scenario == "transient-then-ok":
            if attempt == 1:
                self._send_json(502, _build_score_failed_transient(req_id))
            else:
                self._send_json(200, _build_completed_hot(req_id))
        elif scenario == "partial-retry-then-ok":
            if attempt == 1:
                self._send_json(200, _build_partial_retry(req_id))
            else:
                self._send_json(200, _build_completed_hot(req_id))
        elif scenario == "always-502":
            self._send_json(502, _build_score_failed_transient(req_id))
        elif scenario == "repair-then-ok":
            if attempt == 1:
                self._send_json(200, _build_repair_then_ok(req_id))
            else:
                self._send_json(200, _build_completed_hot(req_id))
        elif scenario == "repair-and-retry":
            if attempt == 1:
                self._send_json(200, _build_repair_and_retry(req_id))
            else:
                self._send_json(200, _build_completed_hot(req_id))
        elif scenario == "repair-fail":
            if attempt == 1:
                self._send_json(200, _build_repair_then_ok(req_id))
            else:
                self._send_json(200, _build_completed_hot(req_id))
        elif scenario == "terminal-config":
            self._send_json(500, _build_score_failed_config(req_id))
        elif scenario == "partial-not-retryable":
            self._send_json(200, _build_partial_not_retryable(req_id))
        elif scenario == "validation-422":
            self._send_json(
                422,
                {
                    "detail": [
                        {
                            "loc": ["body", "lead", "email"],
                            "msg": "value is not a valid email address",
                            "input": lead.get("email"),
                        }
                    ]
                },
            )


class MockServer(ThreadingHTTPServer):
    """Threading HTTP server with attached ServerState."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.state = ServerState()


def make_server(host="0.0.0.0", port=8001):
    """Create and return a configured MockServer instance."""
    return MockServer((host, port), MockHandler)


def main():
    parser = argparse.ArgumentParser(
        description="n8n mock backend and Slack server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host interface to bind"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind"
    )
    args = parser.parse_args()

    server = make_server(args.host, args.port)
    host_str = str(args.host)
    port_str = str(args.port)
    print("=" * 60)
    print("n8n Mock Server running on " + host_str + ":" + port_str)
    print("WARNING: This is a test double with no authentication.")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
