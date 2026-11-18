"""Tests for tools/n8n_mock_server.py.

Exercises every scenario, validation branch, log structure, and error mode
of the mock server test double.
"""

import http.client
import importlib.util
import json
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request
import uuid

# Import tools/n8n_mock_server.py dynamically from file location
_SERVER_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "n8n_mock_server.py"
)
_spec = importlib.util.spec_from_file_location(
    "n8n_mock_server", str(_SERVER_PATH)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_server = _mod.make_server


class TestMockServer(unittest.TestCase):
    """Test suite for the n8n mock backend and Slack server."""

    def setUp(self):
        self.server = make_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)

    def _request(self, method, path, body=None, headers=None):
        url = "http://127.0.0.1:" + str(self.port) + path
        data = None
        hdrs = dict(headers or {})
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode("utf-8")
                if "Content-Type" not in hdrs:
                    hdrs["Content-Type"] = "application/json"
            elif isinstance(body, str):
                data = body.encode("utf-8")
            elif isinstance(body, bytes):
                data = body
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            try:
                return err.code, err.read().decode("utf-8")
            finally:
                err.close()

    def _make_lead_payload(self, scenario=None, lead_id=None, email=None):
        uid = lead_id or str(uuid.uuid4())
        msg = "Need a 3-bedroom house in Austin."
        if scenario:
            msg = msg + " [scenario:" + scenario + "]"
        mail = email or "buyer@example.com"
        return {
            "id": uid,
            "lead": {
                "name": "Alex Mercer",
                "email": mail,
                "phone": "+15125550199",
                "message": msg,
            },
        }

    def test_health(self):
        status, body = self._request("GET", "/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data.get("status"), "ok")

    def test_scenarios(self):
        scenarios_to_test = [
            ("completed-hot", 200, "completed", "hot", 87, False, None),
            ("completed-warm", 200, "completed", "warm", 55, False, None),
            ("completed-cold", 200, "completed", "cold", 20, False, None),
            ("completed-dry-run", 200, "completed", "hot", 87, False, None),
            ("terminal-config", 500, "failed", None, None, False, "score"),
            ("partial-not-retryable", 200, "partial", "hot", 87, False, "crm"),
            ("slow-1", 200, "completed", "hot", 87, False, None),
        ]

        for (
            scen,
            exp_http,
            exp_status,
            exp_cat,
            exp_score,
            exp_retry,
            exp_failed,
        ) in scenarios_to_test:
            with self.subTest(scenario=scen):
                uid = str(uuid.uuid4())
                payload = self._make_lead_payload(scen, lead_id=uid)
                status, body = self._request("POST", "/leads/process", payload)
                self.assertEqual(status, exp_http)
                data = json.loads(body)
                self.assertEqual(data.get("lead_id"), uid)
                self.assertEqual(data.get("status"), exp_status)
                self.assertEqual(data.get("category"), exp_cat)
                self.assertEqual(data.get("score"), exp_score)
                self.assertEqual(len(data.get("steps", [])), 6)
                self.assertEqual(
                    data.get("retry", {}).get("recommended"), exp_retry
                )
                self.assertEqual(data.get("failed_step"), exp_failed)

        # completed-cold step details
        p_cold = self._make_lead_payload("completed-cold")
        st, body = self._request("POST", "/leads/process", p_cold)
        self.assertEqual(st, 200)
        d_cold = json.loads(body)
        skipped = [s["name"] for s in d_cold["steps"] if s["status"] == "skipped"]
        self.assertEqual(skipped, ["email", "sms", "crm"])
        for s in d_cold["steps"]:
            if s["name"] in ("email", "sms", "crm"):
                self.assertEqual(s["reason"], "cold_lead")

        # completed-dry-run step details
        p_dry = self._make_lead_payload("completed-dry-run")
        st, body = self._request("POST", "/leads/process", p_dry)
        self.assertEqual(st, 200)
        d_dry = json.loads(body)
        for s in d_dry["steps"]:
            if s["name"] in ("email", "sms", "crm"):
                self.assertEqual(s["status"], "skipped")
                self.assertEqual(s["reason"], "dry_run")

        # always-502 scenario
        p_502 = self._make_lead_payload("always-502")
        st1, b1 = self._request("POST", "/leads/process", p_502)
        st2, b2 = self._request("POST", "/leads/process", p_502)
        self.assertEqual(st1, 502)
        self.assertEqual(st2, 502)
        d1 = json.loads(b1)
        self.assertEqual(d1["status"], "failed")
        self.assertEqual(d1["failed_step"], "score")
        self.assertEqual(d1["retry"]["recommended"], True)
        self.assertEqual(d1["retry"]["steps"], ["score"])

        # partial-retry-then-ok scenario
        p_pr = self._make_lead_payload("partial-retry-then-ok")
        st_pr1, b_pr1 = self._request("POST", "/leads/process", p_pr)
        self.assertEqual(st_pr1, 200)
        d_pr1 = json.loads(b_pr1)
        self.assertEqual(d_pr1["status"], "partial")
        self.assertEqual(d_pr1["failed_step"], "sms")
        self.assertEqual(d_pr1["retry"]["recommended"], True)
        self.assertEqual(d_pr1["retry"]["steps"], ["sms"])
        st_pr2, b_pr2 = self._request("POST", "/leads/process", p_pr)
        self.assertEqual(st_pr2, 200)
        d_pr2 = json.loads(b_pr2)
        self.assertEqual(d_pr2["status"], "completed")

        # repair-then-ok scenario
        p_rep = self._make_lead_payload("repair-then-ok")
        st_rep1, b_rep1 = self._request("POST", "/leads/process", p_rep)
        self.assertEqual(st_rep1, 200)
        d_rep1 = json.loads(b_rep1)
        self.assertEqual(d_rep1["status"], "partial")
        self.assertEqual(d_rep1["failed_step"], "email")
        self.assertEqual(len(d_rep1["repair"]), 1)
        exp_act = "PATCH /leads/" + p_rep["id"] + "/flags"
        self.assertEqual(d_rep1["repair"][0]["action"], exp_act)
        self.assertEqual(
            d_rep1["repair"][0]["body"], {"email_sent": True}
        )
        st_rep2, b_rep2 = self._request("POST", "/leads/process", p_rep)
        self.assertEqual(st_rep2, 200)
        d_rep2 = json.loads(b_rep2)
        self.assertEqual(d_rep2["status"], "completed")

        # repair-and-retry scenario
        p_rr = self._make_lead_payload("repair-and-retry")
        st_rr1, b_rr1 = self._request("POST", "/leads/process", p_rr)
        self.assertEqual(st_rr1, 200)
        d_rr1 = json.loads(b_rr1)
        self.assertEqual(d_rr1["status"], "partial")
        self.assertEqual(len(d_rr1["repair"]), 1)
        self.assertEqual(d_rr1["retry"]["recommended"], True)
        self.assertEqual(d_rr1["retry"]["steps"], ["sms"])
        st_rr2, b_rr2 = self._request("POST", "/leads/process", p_rr)
        self.assertEqual(st_rr2, 200)
        d_rr2 = json.loads(b_rr2)
        self.assertEqual(d_rr2["status"], "completed")

        # repair-fail scenario
        p_rf = self._make_lead_payload("repair-fail")
        st_rf1, b_rf1 = self._request("POST", "/leads/process", p_rf)
        self.assertEqual(st_rf1, 200)
        d_rf1 = json.loads(b_rf1)
        self.assertEqual(d_rf1["status"], "partial")
        st_rf2, b_rf2 = self._request("POST", "/leads/process", p_rf)
        self.assertEqual(st_rf2, 200)
        d_rf2 = json.loads(b_rf2)
        self.assertEqual(d_rf2["status"], "completed")

        # validation-422 scenario
        p_v422 = self._make_lead_payload(
            "validation-422", email="bad_email_format"
        )
        st_v, b_v = self._request("POST", "/leads/process", p_v422)
        self.assertEqual(st_v, 422)
        d_v = json.loads(b_v)
        self.assertIn("detail", d_v)
        self.assertEqual(d_v["detail"][0]["input"], "bad_email_format")

    def test_per_id_call_counting(self):
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())

        p_a = self._make_lead_payload("transient-then-ok", lead_id=id_a)
        p_b = self._make_lead_payload("transient-then-ok", lead_id=id_b)

        # Call 1 for id_a returns 502
        st_a1, b_a1 = self._request("POST", "/leads/process", p_a)
        self.assertEqual(st_a1, 502)
        d_a1 = json.loads(b_a1)
        self.assertEqual(d_a1["status"], "failed")

        # Call 2 for id_a returns 200
        st_a2, b_a2 = self._request("POST", "/leads/process", p_a)
        self.assertEqual(st_a2, 200)
        d_a2 = json.loads(b_a2)
        self.assertEqual(d_a2["status"], "completed")

        # Call 1 for id_b starts at call 1 -> returns 502
        st_b1, b_b1 = self._request("POST", "/leads/process", p_b)
        self.assertEqual(st_b1, 502)
        d_b1 = json.loads(b_b1)
        self.assertEqual(d_b1["status"], "failed")

    def test_patch_recording_and_repair_fail_500(self):
        id_fail = str(uuid.uuid4())
        p_fail = self._make_lead_payload("repair-fail", lead_id=id_fail)
        st1, _ = self._request("POST", "/leads/process", p_fail)
        self.assertEqual(st1, 200)

        # PATCH for id_fail must answer 500 {"detail": "boom"}
        patch_path_fail = "/leads/" + id_fail + "/flags"
        st_patch_fail, b_patch_fail = self._request(
            "PATCH", patch_path_fail, {"email_sent": True}
        )
        self.assertEqual(st_patch_fail, 500)
        d_patch_fail = json.loads(b_patch_fail)
        self.assertEqual(d_patch_fail.get("detail"), "boom")

        # PATCH for normal id must answer 200 {"id": id, "ok": True}
        id_ok = str(uuid.uuid4())
        p_ok = self._make_lead_payload("repair-then-ok", lead_id=id_ok)
        st2, _ = self._request("POST", "/leads/process", p_ok)
        self.assertEqual(st2, 200)

        patch_path_ok = "/leads/" + id_ok + "/flags"
        st_patch_ok, b_patch_ok = self._request(
            "PATCH", patch_path_ok, {"email_sent": True}
        )
        self.assertEqual(st_patch_ok, 200)
        d_patch_ok = json.loads(b_patch_ok)
        self.assertEqual(d_patch_ok.get("id"), id_ok)
        self.assertEqual(d_patch_ok.get("ok"), True)

        # Check that both PATCH events were recorded in _log
        st_log, b_log = self._request("GET", "/_log")
        self.assertEqual(st_log, 200)
        events = json.loads(b_log)
        patches = [e for e in events if e.get("kind") == "patch"]
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0]["id"], id_fail)
        self.assertEqual(patches[0]["payload"], {"email_sent": True})
        self.assertEqual(patches[1]["id"], id_ok)
        self.assertEqual(patches[1]["payload"], {"email_sent": True})

    def test_slack_sinks_record_bodies(self):
        lead_payload = {"text": "Hot lead detected: score 87"}
        alert_payload = {"text": "Pipeline failure: retries exhausted"}

        st1, b1 = self._request("POST", "/slack/leads", lead_payload)
        self.assertEqual(st1, 200)
        self.assertEqual(b1, "ok")

        st2, b2 = self._request("POST", "/slack/alerts", alert_payload)
        self.assertEqual(st2, 200)
        self.assertEqual(b2, "ok")

        st_log, b_log = self._request("GET", "/_log")
        self.assertEqual(st_log, 200)
        events = json.loads(b_log)

        slack_leads_events = [
            e for e in events if e.get("kind") == "slack_leads"
        ]
        slack_alerts_events = [
            e for e in events if e.get("kind") == "slack_alerts"
        ]

        self.assertEqual(len(slack_leads_events), 1)
        self.assertEqual(slack_leads_events[0]["payload"], lead_payload)
        self.assertEqual(len(slack_alerts_events), 1)
        self.assertEqual(slack_alerts_events[0]["payload"], alert_payload)

    def test_log_order_and_content(self):
        uid = str(uuid.uuid4())
        p = self._make_lead_payload("completed-warm", lead_id=uid)

        self._request("POST", "/leads/process", p)
        self._request("POST", "/slack/leads", {"text": "summary"})
        self._request("PATCH", "/leads/" + uid + "/flags", {"sms_sent": True})
        self._request("POST", "/slack/alerts", {"text": "alert"})

        st_log, b_log = self._request("GET", "/_log")
        self.assertEqual(st_log, 200)
        events = json.loads(b_log)
        self.assertEqual(len(events), 4)

        # Order must be strictly 1, 2, 3, 4
        seqs = [e["seq"] for e in events]
        self.assertEqual(seqs, [1, 2, 3, 4])

        kinds = [e["kind"] for e in events]
        self.assertEqual(
            kinds, ["process", "slack_leads", "patch", "slack_alerts"]
        )

        proc_ev = events[0]
        self.assertEqual(proc_ev["id"], uid)
        self.assertEqual(proc_ev["scenario"], "completed-warm")
        self.assertEqual(proc_ev["attempt"], 1)
        self.assertIsNone(proc_ev["payload"])

        # Process event must NEVER store the lead
        serialized_proc = json.dumps(proc_ev)
        self.assertNotIn("Alex Mercer", serialized_proc)
        self.assertNotIn("buyer@example.com", serialized_proc)
        self.assertNotIn("+15125550199", serialized_proc)

    def test_422_echoes_input_while_log_does_not(self):
        canary = "canary_private_data@example.com"
        payload = self._make_lead_payload("validation-422", email=canary)

        st, body = self._request("POST", "/leads/process", payload)
        self.assertEqual(st, 422)
        data = json.loads(body)
        self.assertIn("detail", data)
        self.assertEqual(data["detail"][0]["input"], canary)

        # GET /_log must not contain the canary input
        st_log, log_body = self._request("GET", "/_log")
        self.assertEqual(st_log, 200)
        self.assertNotIn(canary, log_body)

    def test_unknown_scenario_400(self):
        payload = self._make_lead_payload("unregistered-scenario-xyz")
        st, body = self._request("POST", "/leads/process", payload)
        self.assertEqual(st, 400)
        data = json.loads(body)
        self.assertEqual(data.get("detail"), "unknown scenario")

    def test_missing_field_validation_422(self):
        # Missing id
        st1, b1 = self._request(
            "POST",
            "/leads/process",
            {"lead": {"name": "A", "email": "a@b.com", "phone": "1", "message": "m"}},
        )
        self.assertEqual(st1, 422)
        d1 = json.loads(b1)
        locs1 = [err["loc"] for err in d1["detail"]]
        self.assertIn(["body", "id"], locs1)

        # Invalid UUID id
        st2, b2 = self._request(
            "POST",
            "/leads/process",
            {
                "id": "not-a-valid-uuid",
                "lead": {"name": "A", "email": "a@b.com", "phone": "1", "message": "m"},
            },
        )
        self.assertEqual(st2, 422)
        d2 = json.loads(b2)
        self.assertEqual(d2["detail"][0]["msg"], "value is not a valid uuid")

        # Missing name in lead
        uid = str(uuid.uuid4())
        st3, b3 = self._request(
            "POST",
            "/leads/process",
            {
                "id": uid,
                "lead": {
                    "email": "test@example.com",
                    "phone": "+123",
                    "message": "hello",
                },
            },
        )
        self.assertEqual(st3, 422)
        d3 = json.loads(b3)
        locs3 = [err["loc"] for err in d3["detail"]]
        self.assertIn(["body", "lead", "name"], locs3)

    def test_drop_connection(self):
        payload = self._make_lead_payload("drop-connection")
        data = json.dumps(payload).encode("utf-8")
        url = "http://127.0.0.1:" + str(self.port) + "/leads/process"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises((http.client.RemoteDisconnected, ConnectionResetError)):
            urllib.request.urlopen(req, timeout=2)

    def test_reset(self):
        p = self._make_lead_payload("completed-hot")
        self._request("POST", "/leads/process", p)
        self._request("POST", "/slack/leads", {"text": "msg"})

        st_log, b_log = self._request("GET", "/_log")
        self.assertEqual(st_log, 200)
        self.assertGreater(len(json.loads(b_log)), 0)

        st_rst, b_rst = self._request("POST", "/_reset")
        self.assertEqual(st_rst, 200)

        st_log2, b_log2 = self._request("GET", "/_log")
        self.assertEqual(st_log2, 200)
        self.assertEqual(json.loads(b_log2), [])

        # Call counters are cleared: transient-then-ok starts at attempt 1 again
        uid = p["id"]
        p_retry = self._make_lead_payload("transient-then-ok", lead_id=uid)
        st_first, _ = self._request("POST", "/leads/process", p_retry)
        self.assertEqual(st_first, 502)


if __name__ == "__main__":
    unittest.main()
