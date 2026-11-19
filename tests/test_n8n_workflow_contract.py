# tests/test_n8n_workflow_contract.py
# Standard-library only: unittest, json, re, pathlib, shutil, subprocess, tempfile
# Run with: python -m unittest discover -s tests -t . -v
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout (all paths relative to this file)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_WORKFLOW_PATH = _REPO / "n8n-workflows" / "lead_pipeline.json"
_COMPOSE_PATH = _REPO / "n8n-workflows" / "docker-compose.yml"
_ENV_EXAMPLE_PATH = _REPO / "n8n-workflows" / ".env.n8n.example"
_GITIGNORE_PATH = _REPO / ".gitignore"

_REQUIRED_FILES = {
    "n8n-workflows/lead_pipeline.json": _WORKFLOW_PATH,
    "n8n-workflows/docker-compose.yml": _COMPOSE_PATH,
    "n8n-workflows/.env.n8n.example": _ENV_EXAMPLE_PATH,
    ".gitignore": _GITIGNORE_PATH,
}

# ---------------------------------------------------------------------------
# Fail fast if any required file is missing
# ---------------------------------------------------------------------------
for _label, _p in _REQUIRED_FILES.items():
    if not _p.is_file():
        raise FileNotFoundError(
            f"Required file not found: {_label!r} (resolved to {_p}). "
            "Make sure to run tests from the repository root."
        )

# ---------------------------------------------------------------------------
# Helpers loaded once at module level
# ---------------------------------------------------------------------------
_RAW_WORKFLOW = _WORKFLOW_PATH.read_text(encoding="utf-8")
_WF = json.loads(_RAW_WORKFLOW)
_NODES: list = _WF["nodes"]
_CONNECTIONS: dict = _WF["connections"]

_RAW_COMPOSE = _COMPOSE_PATH.read_text(encoding="utf-8")
_RAW_ENV_EXAMPLE = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
_RAW_GITIGNORE = _GITIGNORE_PATH.read_text(encoding="utf-8")


def node_by_name(name: str) -> dict:
    for n in _NODES:
        if n.get("name") == name:
            return n
    raise KeyError(f"Node not found: {name!r}")


def jscode(name: str) -> str:
    n = node_by_name(name)
    return n.get("parameters", {}).get("jsCode", "")


def _targets_of(source: str, output_index: int = 0) -> list:
    """Return list of target node names for a given source and output index."""
    conn = _CONNECTIONS.get(source, {})
    outputs = conn.get("main", [])
    if output_index >= len(outputs):
        return []
    return [c["node"] for c in outputs[output_index]]


def _all_targets(source: str) -> list:
    """Return all target node names across all output slots."""
    conn = _CONNECTIONS.get(source, {})
    result = []
    for slot in conn.get("main", []):
        for c in slot:
            result.append(c["node"])
    return result


def _reachable_from(start: str) -> set:
    """BFS from start, return set of reachable node names."""
    visited = set()
    queue = [start]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for tgt in _all_targets(cur):
            queue.append(tgt)
    return visited


def _detect_cycles(connections: dict) -> list:
    """Return list of back-edge (src, tgt) pairs that form cycles."""
    all_nodes = set(n["name"] for n in _NODES)
    adj: dict = {name: [] for name in all_nodes}
    for src, conn in connections.items():
        for slot in conn.get("main", []):
            for c in slot:
                adj[src].append(c["node"])

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in all_nodes}
    cycles = []

    def dfs(u):
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == GRAY:
                cycles.append((u, v))
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for node in all_nodes:
        if color[node] == WHITE:
            dfs(node)
    return cycles


def _node_can_reach(start: str, target: str, via: str = None) -> bool:
    """Return True if 'start' can reach 'target' (optionally only through 'via')."""
    adj: dict = {}
    for src, conn in _CONNECTIONS.items():
        adj[src] = []
        for slot in conn.get("main", []):
            for c in slot:
                adj[src].append(c["node"])
    visited = set()
    queue = [start]
    first = True
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        if not first and cur == target:
            return True
        visited.add(cur)
        first = False
        if via is None or cur == start or cur == via or via in _reachable_from(start):
            for tgt in adj.get(cur, []):
                queue.append(tgt)
    return False


# ---------------------------------------------------------------------------
# Small JS bracket-balance checker (strips strings, templates, regex, comments)
# ---------------------------------------------------------------------------
def _check_bracket_balance(js: str) -> str | None:
    """
    Return None if balanced, or an error string.
    Strips single-quoted, double-quoted, template-literal strings, block/line
    comments, and regex literals (preceded by operator/keyword context) before
    counting (), {}, [].
    """
    # Tokenise: produce a reduced string with only structural characters
    i = 0
    buf = []
    n = len(js)
    # Helpers: set of chars that can precede a regex literal
    _REGEX_PRECEDE = set("=({[,!&|?:+-*/%^~;<>")

    def prev_non_ws(b):
        for ch in reversed(b):
            if ch not in (" ", "\t", "\n", "\r"):
                return ch
        return ""

    while i < n:
        c = js[i]

        # Line comment
        if c == "/" and i + 1 < n and js[i + 1] == "/":
            while i < n and js[i] != "\n":
                i += 1
            continue

        # Block comment
        if c == "/" and i + 1 < n and js[i + 1] == "*":
            i += 2
            while i + 1 < n and not (js[i] == "*" and js[i + 1] == "/"):
                i += 1
            i += 2
            continue

        # Template literal `...`
        if c == "`":
            i += 1
            depth = 1
            while i < n and depth > 0:
                if js[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if js[i] == "`":
                    depth -= 1
                i += 1
            continue

        # Double-quoted string
        if c == '"':
            i += 1
            while i < n and js[i] != '"':
                if js[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            i += 1
            continue

        # Single-quoted string
        if c == "'":
            i += 1
            while i < n and js[i] != "'":
                if js[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                i += 1
            i += 1
            continue

        # Regex literal: / that follows an operator or open-bracket context
        if c == "/" and i + 1 < n and js[i + 1] not in ("*", "/"):
            prev = prev_non_ws(buf)
            if prev in _REGEX_PRECEDE or prev == "":
                # Consume regex
                i += 1
                in_cls = False
                while i < n:
                    ch = js[i]
                    if ch == "\\" and i + 1 < n:
                        i += 2
                        continue
                    if ch == "[":
                        in_cls = True
                    elif ch == "]":
                        in_cls = False
                    elif ch == "/" and not in_cls:
                        i += 1
                        # skip flags
                        while i < n and js[i].isalpha():
                            i += 1
                        break
                    i += 1
                continue

        buf.append(c)
        i += 1

    reduced = "".join(buf)
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    openers = set("({[")
    closers = set(")}]")
    for ch in reduced:
        if ch in openers:
            stack.append(ch)
        elif ch in closers:
            if not stack or stack[-1] != pairs[ch]:
                return f"Unmatched '{ch}' (stack: {stack[-5:]})"
            stack.pop()
    if stack:
        return f"Unclosed brackets: {stack}"
    return None


# ===========================================================================
# A. Structure
# ===========================================================================
class TestAStructure(unittest.TestCase):

    def test_a01_valid_json(self):
        # Already parsed at module level; just assert the global is a dict
        self.assertIsInstance(_WF, dict, "lead_pipeline.json did not parse as a JSON object")

    def test_a02_required_top_level_keys(self):
        for key in ("name", "nodes", "connections", "settings"):
            self.assertIn(key, _WF, f"Missing top-level key: {key!r}")

    def test_a03_banned_top_level_keys(self):
        for key in ("pinData", "staticData", "id", "versionId"):
            self.assertNotIn(key, _WF, f"Banned top-level key present: {key!r}")

    def test_a04_meta_no_instance_id(self):
        if "meta" in _WF:
            self.assertNotIn(
                "instanceId", _WF["meta"],
                "Top-level 'meta' must not contain 'instanceId'"
            )

    def test_a05_active_is_false(self):
        self.assertIs(_WF.get("active"), False, "'active' must be false")

    def test_a06_node_names_unique(self):
        names = [n["name"] for n in _NODES]
        seen = set()
        dupes = [nm for nm in names if nm in seen or seen.add(nm)]
        self.assertEqual(dupes, [], f"Duplicate node names: {dupes}")

    def test_a07_node_ids_unique_strings(self):
        ids = [n.get("id") for n in _NODES]
        for nid in ids:
            self.assertIsInstance(nid, str, f"Node id is not a string: {nid!r}")
        seen = set()
        dupes = [nid for nid in ids if nid in seen or seen.add(nid)]
        self.assertEqual(dupes, [], f"Duplicate node ids: {dupes}")

    def test_a08_every_node_has_valid_type(self):
        for n in _NODES:
            self.assertTrue(
                isinstance(n.get("type"), str) and n["type"].startswith("n8n-nodes-base."),
                f"Node {n.get('name')!r} has invalid type: {n.get('type')!r}"
            )

    def test_a09_every_node_has_numeric_type_version(self):
        for n in _NODES:
            tv = n.get("typeVersion")
            self.assertIsInstance(
                tv, (int, float),
                f"Node {n.get('name')!r} typeVersion is not numeric: {tv!r}"
            )

    def test_a10_every_node_has_valid_position(self):
        for n in _NODES:
            pos = n.get("position")
            self.assertIsInstance(pos, list, f"Node {n.get('name')!r}: position is not a list")
            self.assertEqual(len(pos), 2, f"Node {n.get('name')!r}: position must have 2 elements")
            for coord in pos:
                self.assertIsInstance(
                    coord, (int, float),
                    f"Node {n.get('name')!r}: position coordinate {coord!r} is not a number"
                )

    def test_a11_every_node_has_parameters_dict(self):
        for n in _NODES:
            self.assertIsInstance(
                n.get("parameters"), dict,
                f"Node {n.get('name')!r}: 'parameters' is missing or not a dict"
            )

    def test_a12_no_credentials_key_on_any_node(self):
        for n in _NODES:
            self.assertNotIn(
                "credentials", n,
                f"Node {n.get('name')!r} must not have a 'credentials' key"
            )

    def test_a13_no_webhook_id_on_any_node(self):
        for n in _NODES:
            self.assertNotIn(
                "webhookId", n,
                f"Node {n.get('name')!r} must not have a 'webhookId' key"
            )


# ===========================================================================
# B. Required nodes & type counts
# ===========================================================================
_REQUIRED_NODE_NAMES = [
    "Lead Webhook", "Validate Lead", "Route: Valid?", "Respond Invalid",
    "Init State", "Respond 202", "Prepare Attempt", "Call Pipeline",
    "Evaluate Response", "Route Decision", "Build Repairs", "Apply Repair",
    "After Repair", "Compute Backoff", "Wait Before Retry", "Build Slack Success",
    "Slack: Lead Summary", "Build Slack Alert", "Slack: Error Alert",
    "Fail Execution (keep data)", "Cold Lead: No Alert",
]


class TestBRequiredNodes(unittest.TestCase):

    def test_b01_all_21_names_present(self):
        existing = {n["name"] for n in _NODES}
        for name in _REQUIRED_NODE_NAMES:
            self.assertIn(name, existing, f"Required node missing: {name!r}")

    def test_b02_exactly_21_nodes(self):
        self.assertEqual(len(_NODES), 21, f"Expected 21 nodes, got {len(_NODES)}")

    def test_b03_type_counts(self):
        from collections import Counter
        suffix_counts = Counter(n["type"].split(".")[-1] for n in _NODES)
        self.assertEqual(suffix_counts["webhook"], 1, "Expected 1 webhook node")
        self.assertEqual(suffix_counts["respondToWebhook"], 2, "Expected 2 respondToWebhook nodes")
        self.assertEqual(suffix_counts["httpRequest"], 4, "Expected 4 httpRequest nodes")
        self.assertEqual(suffix_counts["switch"], 2, "Expected 2 switch nodes")
        self.assertEqual(suffix_counts["wait"], 1, "Expected 1 wait node")
        self.assertEqual(suffix_counts["stopAndError"], 1, "Expected 1 stopAndError node")
        self.assertEqual(suffix_counts["noOp"], 1, "Expected 1 noOp node")
        self.assertEqual(suffix_counts["code"], 9, "Expected 9 code nodes")


# ===========================================================================
# C. Webhook node
# ===========================================================================
class TestCWebhook(unittest.TestCase):

    def setUp(self):
        self.node = node_by_name("Lead Webhook")
        self.params = self.node["parameters"]

    def test_c01_method_post(self):
        self.assertEqual(self.params.get("httpMethod"), "POST")

    def test_c02_path_lead_intake(self):
        self.assertEqual(self.params.get("path"), "lead-intake")

    def test_c03_response_mode_via_node(self):
        self.assertEqual(
            self.params.get("responseMode"), "responseNode",
            "Webhook responseMode must be 'responseNode' (respond via a Respond to Webhook node)"
        )


# ===========================================================================
# D. Call Pipeline HTTP node
# ===========================================================================
class TestDCallPipeline(unittest.TestCase):

    def setUp(self):
        self.node = node_by_name("Call Pipeline")
        self.params = self.node["parameters"]
        self.opts = self.params.get("options", {})
        self.resp = self.opts.get("response", {}).get("response", {})

    def test_d01_method_post(self):
        self.assertEqual(self.params.get("method"), "POST")

    def test_d02_url_contains_backend_base_url(self):
        url = self.params.get("url", "")
        self.assertIn("$env.BACKEND_BASE_URL", url,
                      "url must reference $env.BACKEND_BASE_URL")

    def test_d03_url_contains_leads_process(self):
        url = self.params.get("url", "")
        self.assertIn("/leads/process", url,
                      "url must contain '/leads/process'")

    def test_d04_body_contains_id_and_lead(self):
        body = self.params.get("jsonBody", "")
        self.assertIn("id", body, "jsonBody must reference 'id'")
        self.assertIn("lead", body, "jsonBody must reference 'lead'")

    def test_d05_timeout_300000(self):
        self.assertEqual(self.opts.get("timeout"), 300000,
                         "timeout must be exactly 300000 ms")

    def test_d06_full_response_true(self):
        self.assertIs(self.resp.get("fullResponse"), True,
                      "fullResponse must be true")

    def test_d07_never_error_true(self):
        self.assertIs(self.resp.get("neverError"), True,
                      "neverError must be true")

    def test_d08_on_error_continue_regular(self):
        self.assertEqual(self.node.get("onError"), "continueRegularOutput",
                         "onError must be 'continueRegularOutput'")

    def test_d09_no_built_in_retry(self):
        retry = self.params.get("retryOnFail")
        self.assertIsNot(retry, True,
                         "retryOnFail must not be true (retries are handled by the workflow)")


# ===========================================================================
# E. Apply Repair HTTP node
# ===========================================================================
class TestEApplyRepair(unittest.TestCase):

    def setUp(self):
        self.node = node_by_name("Apply Repair")
        self.params = self.node["parameters"]
        self.opts = self.params.get("options", {})
        self.resp = self.opts.get("response", {}).get("response", {})

    def test_e01_method_patch(self):
        self.assertEqual(self.params.get("method"), "PATCH")

    def test_e02_url_contains_backend_base_url(self):
        url = self.params.get("url", "")
        self.assertIn("$env.BACKEND_BASE_URL", url)

    def test_e03_url_contains_patch_url_path(self):
        url = self.params.get("url", "")
        self.assertIn("patchUrlPath", url,
                      "url must reference $json.patchUrlPath")

    def test_e04_body_contains_patch_body(self):
        body = self.params.get("jsonBody", "")
        self.assertIn("patchBody", body,
                      "jsonBody must reference patchBody")

    def test_e05_full_response_true(self):
        self.assertIs(self.resp.get("fullResponse"), True)

    def test_e06_never_error_true(self):
        self.assertIs(self.resp.get("neverError"), True)

    def test_e07_on_error_continue_regular(self):
        self.assertEqual(self.node.get("onError"), "continueRegularOutput")


# ===========================================================================
# F. Slack nodes
# ===========================================================================
class TestFSlackNodes(unittest.TestCase):

    def _resp(self, name):
        n = node_by_name(name)
        return n, n["parameters"], n["parameters"].get("options", {}).get("response", {}).get("response", {})

    def test_f01_leads_url_uses_env(self):
        _, params, _ = self._resp("Slack: Lead Summary")
        self.assertIn("$env.SLACK_LEADS_WEBHOOK_URL", params.get("url", ""))

    def test_f02_alerts_url_uses_env(self):
        _, params, _ = self._resp("Slack: Error Alert")
        self.assertIn("$env.SLACK_ALERTS_WEBHOOK_URL", params.get("url", ""))

    def test_f03_leads_never_error(self):
        _, _, resp = self._resp("Slack: Lead Summary")
        self.assertIs(resp.get("neverError"), True)

    def test_f04_alerts_never_error(self):
        _, _, resp = self._resp("Slack: Error Alert")
        self.assertIs(resp.get("neverError"), True)

    def test_f05_leads_on_error_continue(self):
        node, _, _ = self._resp("Slack: Lead Summary")
        self.assertEqual(node.get("onError"), "continueRegularOutput")

    def test_f06_alerts_on_error_continue(self):
        node, _, _ = self._resp("Slack: Error Alert")
        self.assertEqual(node.get("onError"), "continueRegularOutput")


# ===========================================================================
# G. Settings
# ===========================================================================
class TestGSettings(unittest.TestCase):

    def setUp(self):
        self.s = _WF.get("settings", {})

    def test_g01_execution_order_v1(self):
        self.assertEqual(self.s.get("executionOrder"), "v1")

    def test_g02_execution_timeout_at_least_1500(self):
        t = self.s.get("executionTimeout")
        self.assertIsInstance(t, (int, float))
        self.assertGreaterEqual(t, 1500,
                                "executionTimeout should be >= 1500 seconds")

    def test_g03_save_success_none(self):
        self.assertEqual(self.s.get("saveDataSuccessExecution"), "none")

    def test_g04_save_error_all(self):
        self.assertEqual(self.s.get("saveDataErrorExecution"), "all")


# ===========================================================================
# G2. Wait Before Retry unit
# ===========================================================================
class TestG2WaitBeforeRetry(unittest.TestCase):

    def setUp(self):
        self.node = node_by_name("Wait Before Retry")
        self.params = self.node["parameters"]

    def test_g2_01_amount_references_wait_seconds(self):
        amount = self.params.get("amount", "")
        self.assertIn("waitSeconds", str(amount),
                      "'amount' parameter must reference 'waitSeconds'")

    def test_g2_02_unit_key_exists_and_is_seconds(self):
        self.assertIn(
            "unit", self.params,
            "Wait Before Retry is missing a 'unit' key. Without it n8n defaults to HOURS, "
            "turning the 60/120/240-second backoffs into 60/120/240-hour waits."
        )
        self.assertEqual(
            self.params["unit"], "seconds",
            f"Wait Before Retry unit must be 'seconds', got {self.params['unit']!r}. "
            "Without 'seconds', n8n defaults to hours."
        )


# ===========================================================================
# H. Connections & reachability
# ===========================================================================
class TestHConnections(unittest.TestCase):

    def _assert_edge(self, src, tgt, output_index=0):
        targets = _targets_of(src, output_index)
        self.assertIn(
            tgt, targets,
            f"Expected edge {src!r} [output {output_index}] -> {tgt!r}; "
            f"got targets: {targets}"
        )

    def test_h01_all_connection_sources_are_known_nodes(self):
        node_names = {n["name"] for n in _NODES}
        for src in _CONNECTIONS:
            self.assertIn(src, node_names,
                          f"Connection source {src!r} is not a node name")

    def test_h02_all_connection_targets_are_known_nodes(self):
        node_names = {n["name"] for n in _NODES}
        for src, conn in _CONNECTIONS.items():
            for slot in conn.get("main", []):
                for c in slot:
                    self.assertIn(c["node"], node_names,
                                  f"Connection target {c['node']!r} from {src!r} is not a node name")

    def test_h03_all_nodes_reachable_from_lead_webhook(self):
        reachable = _reachable_from("Lead Webhook")
        for n in _NODES:
            self.assertIn(
                n["name"], reachable,
                f"Node {n['name']!r} is not reachable from 'Lead Webhook'"
            )

    def test_h04_linear_spine(self):
        spine = [
            ("Lead Webhook", "Validate Lead"),
            ("Validate Lead", "Route: Valid?"),
            ("Init State", "Respond 202"),
            ("Respond 202", "Prepare Attempt"),
            ("Prepare Attempt", "Call Pipeline"),
            ("Call Pipeline", "Evaluate Response"),
            ("Evaluate Response", "Route Decision"),
        ]
        for src, tgt in spine:
            self._assert_edge(src, tgt, 0)

    def test_h05_route_valid_outputs(self):
        self._assert_edge("Route: Valid?", "Init State", 0)
        self._assert_edge("Route: Valid?", "Respond Invalid", 1)

    def test_h06_route_decision_5_outputs(self):
        expected = [
            (0, "Build Repairs"),
            (1, "Compute Backoff"),
            (2, "Build Slack Success"),
            (3, "Cold Lead: No Alert"),
            (4, "Build Slack Alert"),
        ]
        for idx, tgt in expected:
            self._assert_edge("Route Decision", tgt, idx)

    def test_h07_repair_loop(self):
        self._assert_edge("Build Repairs", "Apply Repair", 0)
        self._assert_edge("Apply Repair", "After Repair", 0)
        self._assert_edge("After Repair", "Route Decision", 0)

    def test_h08_retry_loop(self):
        self._assert_edge("Compute Backoff", "Wait Before Retry", 0)
        self._assert_edge("Wait Before Retry", "Prepare Attempt", 0)

    def test_h09_alert_terminal_path(self):
        self._assert_edge("Build Slack Alert", "Slack: Error Alert", 0)
        self._assert_edge("Slack: Error Alert", "Fail Execution (keep data)", 0)

    def test_h10_repair_loop_exists(self):
        # After Repair -> Route Decision -> Build Repairs -> Apply Repair -> After Repair
        self.assertIn("Route Decision", _all_targets("After Repair"),
                      "After Repair must connect back to Route Decision (repair loop)")
        self.assertIn("Build Repairs", _targets_of("Route Decision", 0),
                      "Route Decision output 0 must feed Build Repairs (repair loop)")
        self.assertIn("Apply Repair", _targets_of("Build Repairs", 0),
                      "Build Repairs must connect to Apply Repair (repair loop)")
        self.assertIn("After Repair", _targets_of("Apply Repair", 0),
                      "Apply Repair must connect to After Repair (repair loop)")

    def test_h11_retry_loop_exists(self):
        # Compute Backoff -> Wait Before Retry -> Prepare Attempt -> ... -> Route Decision
        self.assertIn("Wait Before Retry", _targets_of("Compute Backoff", 0),
                      "Compute Backoff must connect to Wait Before Retry (retry loop)")
        self.assertIn("Prepare Attempt", _targets_of("Wait Before Retry", 0),
                      "Wait Before Retry must connect back to Prepare Attempt (retry loop)")
        reachable_from_prepare = _reachable_from("Prepare Attempt")
        self.assertIn("Route Decision", reachable_from_prepare,
                      "Prepare Attempt must eventually reach Route Decision (retry loop closure)")
        self.assertIn("Compute Backoff", reachable_from_prepare,
                      "Prepare Attempt must eventually reach Compute Backoff (retry loop closure)")

    def test_h12_no_extra_cycles(self):
        # DFS must find exactly 2 back-edges: one for the repair loop, one for the retry loop.
        # Any additional back-edge means an unintended cycle was introduced.
        cycles = _detect_cycles(_CONNECTIONS)
        self.assertEqual(
            len(cycles), 2,
            f"Expected exactly 2 DFS back-edges (repair + retry loops), found {len(cycles)}: {cycles}"
        )


# ===========================================================================
# I. Sanitisation (whole file text)
# ===========================================================================
_ALLOWED_ENV_NAMES = {
    "BACKEND_BASE_URL", "SLACK_LEADS_WEBHOOK_URL", "SLACK_ALERTS_WEBHOOK_URL",
    "WEBHOOK_SHARED_SECRET", "N8N_EDITOR_BASE_URL",
}


class TestISanitisation(unittest.TestCase):

    def _assert_no_match(self, pattern, flags=0):
        m = re.search(pattern, _RAW_WORKFLOW, flags)
        matched = repr(m.group()) if m else ""
        self.assertIsNone(
            m,
            f"Forbidden pattern {pattern!r} matched in lead_pipeline.json: {matched}"
        )

    def test_i01_no_slack_webhook_url(self):
        self._assert_no_match(r"hooks\.slack\.com")

    def test_i02_no_slack_tokens(self):
        self._assert_no_match(r"xox[abprs]-")

    def test_i03_no_bearer_token(self):
        self._assert_no_match(r"Bearer\s+\S")

    def test_i04_no_openai_key(self):
        self._assert_no_match(r"sk-or-")

    def test_i05_no_hubspot_api(self):
        self._assert_no_match(r"api\.hubapi")

    def test_i06_no_hubapi(self):
        self._assert_no_match(r"hubapi")

    def test_i07_no_localhost(self):
        self._assert_no_match(r"localhost")

    def test_i08_no_host_docker_internal(self):
        self._assert_no_match(r"host\.docker\.internal")

    def test_i09_no_loopback_ip(self):
        self._assert_no_match(r"127\.0\.0\.1")

    def test_i10_no_bare_ip_address(self):
        self._assert_no_match(r"\b\d{1,3}(\.\d{1,3}){3}\b")

    def test_i11_no_port_8000(self):
        self._assert_no_match(r":8000")

    def test_i12_no_port_8001(self):
        self._assert_no_match(r":8001")

    def test_i13_no_email_address(self):
        self._assert_no_match(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

    def test_i14_no_phone_number(self):
        self._assert_no_match(r"\+\d{8,}")

    def test_i15_all_env_names_allowed(self):
        # Find every $env.SOMETHING reference in the file
        for m in re.finditer(r"\$env\.([A-Z0-9_]+)", _RAW_WORKFLOW):
            name = m.group(1)
            self.assertIn(
                name, _ALLOWED_ENV_NAMES,
                f"Unexpected $env.{name} reference -- only {sorted(_ALLOWED_ENV_NAMES)} are allowed"
            )


# ===========================================================================
# J. Code logic markers
# ===========================================================================
class TestJCodeLogicMarkers(unittest.TestCase):

    def _has(self, node_name, marker):
        code = jscode(node_name)
        self.assertIn(marker, code,
                      f"Code of {node_name!r} must contain {marker!r}")

    def test_j01_init_state_uuid(self):
        self._has("Init State", "randomUUID")

    def test_j02_init_state_attempt_1(self):
        self._has("Init State", "attempt: 1")

    def test_j03_init_state_max_attempts_3(self):
        self._has("Init State", "maxAttempts: 3")

    def test_j04_evaluate_retry_obj(self):
        self._has("Evaluate Response", "retryObj.recommended")

    def test_j05_evaluate_repairs(self):
        self._has("Evaluate Response", "repairs")

    def test_j06_evaluate_hot_warm(self):
        self._has("Evaluate Response", "'hot', 'warm'")

    def test_j07_evaluate_terminal_failure(self):
        self._has("Evaluate Response", "terminal_failure")

    def test_j08_evaluate_network_error(self):
        self._has("Evaluate Response", "networkError")

    def test_j09_evaluate_transient_http_codes(self):
        self._has("Evaluate Response", "429, 502, 503, 504")

    def test_j10_evaluate_repair_before_retry(self):
        code = jscode("Evaluate Response")
        idx_repair = code.find("flag_repair_needed")
        idx_retry = code.find("retryable_steps")
        self.assertGreater(idx_repair, -1,
                           "Evaluate Response must contain 'flag_repair_needed'")
        self.assertGreater(idx_retry, -1,
                           "Evaluate Response must contain 'retryable_steps'")
        self.assertLess(idx_repair, idx_retry,
                        "In Evaluate Response, the repair decision ('flag_repair_needed') "
                        "must appear before the retry decision ('retryable_steps') in the code")

    def test_j11_after_repair_no_repair_decision(self):
        code = jscode("After Repair")
        m = re.search(r"decision\s*=\s*'repair'", code)
        self.assertIsNone(m,
                          "After Repair must never assign decision = 'repair'")

    def test_j12_after_repair_contains_repair_failed(self):
        self._has("After Repair", "repair_failed")

    def test_j13_build_repairs_expected_action(self):
        self._has("Build Repairs", "expectedAction")

    def test_j14_compute_backoff_math_pow(self):
        self._has("Compute Backoff", "Math.pow(2")

    def test_j15_compute_backoff_120(self):
        self._has("Compute Backoff", "120")

    def test_j16_compute_backoff_60(self):
        self._has("Compute Backoff", "60")

    def test_j17_validate_lead_missing(self):
        self._has("Validate Lead", "missing")

    def test_j18_validate_lead_secret_header(self):
        self._has("Validate Lead", "x-webhook-secret")

    def test_j19_slack_success_dry_run(self):
        self._has("Build Slack Success", "DRY RUN")

    def test_j20_slack_success_escape(self):
        self._has("Build Slack Success", "&amp;")

    def test_j21_slack_alert_escape(self):
        self._has("Build Slack Alert", "&amp;")

    def test_j22_slack_alert_repair_failed(self):
        self._has("Build Slack Alert", "repair_failed")


# ===========================================================================
# K. Privacy of Slack builders
# ===========================================================================
class TestKPrivacy(unittest.TestCase):

    def _code(self, name):
        return jscode(name)

    def _assert_no_pattern(self, node_name, pattern, flags=re.IGNORECASE):
        code = self._code(node_name)
        m = re.search(pattern, code, flags)
        matched = repr(m.group()) if m else ""
        self.assertIsNone(
            m,
            f"{node_name!r} must not match {pattern!r} -- found: {matched}"
        )

    def test_k01_success_no_lead_pii_fields(self):
        self._assert_no_pattern(
            "Build Slack Success",
            r"\b(lead|l)\.(name|email|phone|message)\b"
        )

    def test_k02_alert_no_lead_pii_fields(self):
        self._assert_no_pattern(
            "Build Slack Alert",
            r"\b(lead|l)\.(name|email|phone|message)\b"
        )

    def test_k03_success_no_detail(self):
        self._assert_no_pattern("Build Slack Success", r"\.detail\b")

    def test_k04_alert_no_detail(self):
        self._assert_no_pattern("Build Slack Alert", r"\.detail\b")

    def test_k05_success_no_body_dot(self):
        self._assert_no_pattern("Build Slack Success", r"body\.")

    def test_k06_alert_no_body_dot(self):
        self._assert_no_pattern("Build Slack Alert", r"body\.")

    def test_k07_alert_does_not_read_s_lead(self):
        self._assert_no_pattern("Build Slack Alert", r"\bs\.lead\b")


# ===========================================================================
# L. JS bracket balance
# ===========================================================================
_CODE_NODES = [n["name"] for n in _NODES if n["type"] == "n8n-nodes-base.code"]
_NODE_HAS_NODE_CLI = shutil.which("node") is not None


class TestLBracketBalance(unittest.TestCase):

    def _test_node(self, name):
        code = jscode(name)
        err = _check_bracket_balance(code)
        self.assertIsNone(err, f"Bracket imbalance in {name!r}: {err}")

    def test_l_validate_lead(self):
        self._test_node("Validate Lead")

    def test_l_init_state(self):
        self._test_node("Init State")

    def test_l_prepare_attempt(self):
        self._test_node("Prepare Attempt")

    def test_l_evaluate_response(self):
        self._test_node("Evaluate Response")

    def test_l_build_repairs(self):
        self._test_node("Build Repairs")

    def test_l_after_repair(self):
        self._test_node("After Repair")

    def test_l_compute_backoff(self):
        self._test_node("Compute Backoff")

    def test_l_build_slack_success(self):
        self._test_node("Build Slack Success")

    def test_l_build_slack_alert(self):
        self._test_node("Build Slack Alert")

    @unittest.skipUnless(_NODE_HAS_NODE_CLI, "node CLI not found -- skipping syntax check")
    def test_l_node_syntax_check_all_code_nodes(self):
        errors = []
        for name in _CODE_NODES:
            code = jscode(name)
            wrapped = "(function(){\n" + code + "\n})"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False, encoding="utf-8"
            ) as f:
                f.write(wrapped)
                tmp_path = f.name
            try:
                result = subprocess.run(
                    ["node", "--check", tmp_path],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    errors.append(f"{name!r}: {result.stderr.strip()}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        self.assertEqual(
            errors, [],
            "node --check failed for:\n" + "\n".join(errors)
        )


# ===========================================================================
# M. Docker files
# ===========================================================================
class TestMDockerFiles(unittest.TestCase):

    def test_m01_compose_block_env_access(self):
        self.assertIn(
            "N8N_BLOCK_ENV_ACCESS_IN_NODE=false", _RAW_COMPOSE,
            "docker-compose.yml must set N8N_BLOCK_ENV_ACCESS_IN_NODE=false"
        )

    def test_m02_compose_localhost_port(self):
        self.assertIn(
            "127.0.0.1:5678:5678", _RAW_COMPOSE,
            "docker-compose.yml must bind port as 127.0.0.1:5678:5678"
        )

    def test_m03_compose_n8n_volume(self):
        self.assertIn(
            "/home/node/.n8n", _RAW_COMPOSE,
            "docker-compose.yml must mount /home/node/.n8n"
        )

    def test_m04_compose_host_docker_internal(self):
        self.assertIn(
            "host.docker.internal:host-gateway", _RAW_COMPOSE,
            "docker-compose.yml must set extra_hosts host.docker.internal:host-gateway"
        )

    def test_m05_compose_backend_base_url(self):
        self.assertIn(
            "BACKEND_BASE_URL=", _RAW_COMPOSE,
            "docker-compose.yml must pass BACKEND_BASE_URL"
        )

    def test_m06_compose_executions_data_prune(self):
        self.assertIn(
            "EXECUTIONS_DATA_PRUNE=true", _RAW_COMPOSE,
            "docker-compose.yml must set EXECUTIONS_DATA_PRUNE=true"
        )

    def test_m07_compose_no_slack_webhook(self):
        self.assertNotIn(
            "hooks.slack.com", _RAW_COMPOSE,
            "docker-compose.yml must not contain a hardcoded Slack webhook URL"
        )

    def test_m08_env_example_slack_leads_empty(self):
        m = re.search(r"^SLACK_LEADS_WEBHOOK_URL=(.*)$", _RAW_ENV_EXAMPLE, re.MULTILINE)
        self.assertIsNotNone(m, ".env.n8n.example must define SLACK_LEADS_WEBHOOK_URL")
        self.assertEqual(
            m.group(1).strip(), "",
            f"SLACK_LEADS_WEBHOOK_URL must be empty in .env.n8n.example, got {m.group(1)!r}"
        )

    def test_m09_env_example_slack_alerts_empty(self):
        m = re.search(r"^SLACK_ALERTS_WEBHOOK_URL=(.*)$", _RAW_ENV_EXAMPLE, re.MULTILINE)
        self.assertIsNotNone(m, ".env.n8n.example must define SLACK_ALERTS_WEBHOOK_URL")
        self.assertEqual(
            m.group(1).strip(), "",
            f"SLACK_ALERTS_WEBHOOK_URL must be empty in .env.n8n.example, got {m.group(1)!r}"
        )

    def test_m10_env_example_webhook_secret_empty(self):
        m = re.search(r"^WEBHOOK_SHARED_SECRET=(.*)$", _RAW_ENV_EXAMPLE, re.MULTILINE)
        self.assertIsNotNone(m, ".env.n8n.example must define WEBHOOK_SHARED_SECRET")
        self.assertEqual(
            m.group(1).strip(), "",
            f"WEBHOOK_SHARED_SECRET must be empty in .env.n8n.example, got {m.group(1)!r}"
        )

    def test_m11_gitignore_env_n8n(self):
        self.assertIn(
            ".env.n8n", _RAW_GITIGNORE,
            ".gitignore must contain '.env.n8n'"
        )

    def test_m12_gitignore_schema_reference(self):
        self.assertIn(
            "_schema_reference.json", _RAW_GITIGNORE,
            ".gitignore must contain '_schema_reference.json'"
        )


class TestNFixes(unittest.TestCase):
    """Verify the four targeted code fixes applied to lead_pipeline.json."""

    def test_n01_fail_execution_error_message(self):
        n = node_by_name("Fail Execution (keep data)")
        msg = n.get("parameters", {}).get("errorMessage", "")
        self.assertIn(
            "$('Route Decision')", msg,
            "errorMessage must reference $('Route Decision') not $json"
        )
        self.assertNotIn(
            "$json.id", msg,
            "errorMessage must not use $json.id (undefined in this node's context)"
        )
        self.assertNotIn(
            "$json.reason", msg,
            "errorMessage must not use $json.reason (undefined in this node's context)"
        )

    def test_n02_slack_alert_http_422_and_index_of(self):
        code = jscode("Build Slack Alert")
        self.assertIn(
            "http_422", code,
            "Build Slack Alert must handle http_422 reason"
        )
        self.assertIn(
            "indexOf('http_')", code,
            "Build Slack Alert must use indexOf to detect http_ prefix"
        )
        self.assertIn(
            "repair_failed", code,
            "Build Slack Alert must still reference repair_failed"
        )

    def test_n03_slack_success_flag_repair_label(self):
        code = jscode("Build Slack Success")
        self.assertIn(
            "sent (flag repaired)", code,
            "Build Slack Success must label flag_update_failed steps as sent (flag repaired)"
        )
        self.assertIn(
            "flag_update_failed", code,
            "Build Slack Success must check for flag_update_failed reason"
        )
        self.assertIn(
            "&amp;", code,
            "Build Slack Success must HTML-escape output"
        )
        self.assertIn(
            "DRY RUN", code,
            "Build Slack Success must retain DRY RUN label"
        )
        self.assertIn(
            "some steps not sent", code,
            "Build Slack Success must contain 'some steps not sent' (mixed-mode dry-run marker)"
        )
        self.assertIn(
            "nothing was sent", code,
            "Build Slack Success must contain 'nothing was sent' (all-dry-run marker)"
        )
        self.assertIn(
            "dryCount", code,
            "Build Slack Success must compute dryCount per-step rather than reading s.dryRun"
        )
        self.assertNotIn(
            "s.dryRun", code,
            "Build Slack Success must NOT read s.dryRun (replaced by per-step dryCount logic)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
