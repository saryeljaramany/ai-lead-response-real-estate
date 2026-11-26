"""Tests for deployment configuration contracts.

Validates Dockerfile, .dockerignore, .gitignore, Caddyfile, docker-compose.yml,
and example environment configuration files using standard library only.
"""

from pathlib import Path
import re
import unittest

_REPO = Path(__file__).resolve().parent.parent

_REQUIRED_FILES = [
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    "deploy/docker-compose.yml",
    "deploy/Caddyfile",
    "deploy/caddy/Dockerfile",
    "deploy/caddy.env.example",
    "deploy/backend.env.example",
    "deploy/n8n.env.example",
]

for _rel in _REQUIRED_FILES:
    _p = _REPO / _rel
    if not _p.is_file():
        raise FileNotFoundError(
            f"Required deploy file missing: {_rel} (resolved to {_p})"
        )


def strip_comments(text: str) -> str:
    """Remove comment lines and trailing comments outside quotes."""
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "#" in line:
            in_single = False
            in_double = False
            cut = -1
            for i, ch in enumerate(line):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == "#" and not in_single and not in_double:
                    cut = i
                    break
            if cut != -1:
                line = line[:cut].rstrip()
        out.append(line)
    return "\n".join(out)


def parse_compose_services(compose_text: str) -> dict[str, str]:
    """Extract services indented by 2 spaces under the top-level services: block."""
    stripped = strip_comments(compose_text)
    in_services = False
    services: dict[str, str] = {}
    current_svc = None
    current_lines: list[str] = []

    for line in stripped.splitlines():
        if not in_services:
            if line.strip() == "services:":
                in_services = True
            continue

        if line and not line.startswith(" "):
            if current_svc:
                services[current_svc] = "\n".join(current_lines)
            break

        m = re.match(r"^ {2}([a-zA-Z0-9_\-]+):\s*$", line)
        if m:
            if current_svc:
                services[current_svc] = "\n".join(current_lines)
            current_svc = m.group(1)
            current_lines = [line]
        elif current_svc:
            current_lines.append(line)

    if current_svc and current_svc not in services:
        services[current_svc] = "\n".join(current_lines)

    return services


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from an env file, ignoring blanks and comments."""
    res = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            res[k.strip()] = v.strip()
    return res


class TestADockerfile(unittest.TestCase):
    """Assertions for repository root Dockerfile."""

    def setUp(self):
        self.raw = (_REPO / "Dockerfile").read_text(encoding="utf-8")
        self.stripped = strip_comments(self.raw)

    def test_a01_python_version_default_312_or_higher(self):
        m = re.search(r"^ARG\s+PYTHON_VERSION\s*=\s*([0-9.]+)", self.raw, re.MULTILINE)
        self.assertIsNotNone(m, "ARG PYTHON_VERSION not found in Dockerfile")
        version = float(m.group(1))
        self.assertGreaterEqual(
            version,
            3.12,
            f"PYTHON_VERSION default must be 3.12 or higher, got {version}",
        )

    def test_a02_base_image_python_version_slim(self):
        self.assertRegex(
            self.raw,
            r"(?m)^FROM\s+python:\$\{PYTHON_VERSION\}-slim\s+AS\s+base\b",
            "base image must be python:${PYTHON_VERSION}-slim AS base",
        )

    def test_a03_stages_and_runtime_is_last(self):
        stages = re.findall(r"(?m)^FROM\s+\S+\s+AS\s+(\w+)", self.raw)
        self.assertIn("base", stages, "stage 'base' missing")
        self.assertIn("test", stages, "stage 'test' missing")
        self.assertIn("runtime", stages, "stage 'runtime' missing")
        self.assertEqual(
            stages[-1],
            "runtime",
            f"stage 'runtime' must stay LAST, got {stages[-1]!r}",
        )

    def test_a04_runtime_copies_only_application_code(self):
        parts = re.split(r"(?m)^FROM\s+", self.raw)
        stage_map = {}
        for p in parts[1:]:
            m = re.match(r"\S+\s+AS\s+(\w+)", p)
            if m:
                stage_map[m.group(1)] = p

        runtime_content = stage_map["runtime"]
        test_content = stage_map["test"]

        copies = [line.strip() for line in re.findall(r"(?m)^COPY\s+(.+)$", runtime_content)]
        expected = ["main.py ./", "models ./models", "routers ./routers", "services ./services"]
        self.assertEqual(
            sorted(copies),
            sorted(expected),
            f"runtime stage must copy only application code, got: {copies}",
        )
        self.assertNotIn("COPY . .", runtime_content)
        self.assertIn("COPY . .", test_content)

    def test_a05_no_copy_of_env_or_token(self):
        self.assertNotRegex(
            self.raw,
            r"(?m)^COPY\s+.*(?:\.env|token\.json)",
            "Dockerfile must not COPY .env or token.json",
        )

    def test_a06_user_appuser_uid_10001_before_cmd(self):
        self.assertRegex(
            self.raw,
            r"useradd\s+.*--uid\s+10001\b.*appuser",
            "appuser must be created with fixed uid 10001",
        )
        parts = re.split(r"(?m)^FROM\s+", self.raw)
        runtime_content = [p for p in parts if "AS runtime" in p][0]
        user_idx = runtime_content.find("USER appuser")
        cmd_idx = runtime_content.find("CMD ")
        self.assertGreater(user_idx, -1, "USER appuser must exist in runtime stage")
        self.assertGreater(cmd_idx, -1, "CMD must exist in runtime stage")
        self.assertLess(user_idx, cmd_idx, "USER appuser must appear before CMD")

    def test_a07_healthcheck_targets_health(self):
        self.assertRegex(
            self.raw,
            r"HEALTHCHECK\b[\s\S]*?/health",
            "HEALTHCHECK must exist and target /health",
        )

    def test_a08_expose_8000(self):
        self.assertRegex(
            self.raw,
            r"(?m)^EXPOSE\s+8000\b",
            "Dockerfile must contain EXPOSE 8000",
        )

    def test_a09_cmd_uses_uvicorn_host_0000(self):
        parts = re.split(r"(?m)^FROM\s+", self.raw)
        runtime_content = [p for p in parts if "AS runtime" in p][0]
        self.assertRegex(
            runtime_content,
            r'CMD\s*\[.*"uvicorn".*"--host",\s*"0\.0\.0\.0".*\]',
            "Runtime CMD must use uvicorn with --host 0.0.0.0",
        )

    def test_a10_test_stage_runs_unittest_discover(self):
        parts = re.split(r"(?m)^FROM\s+", self.raw)
        test_content = [p for p in parts if "AS test" in p][0]
        self.assertRegex(
            test_content,
            r'CMD\s*\[.*"unittest",\s*"discover".*\]',
            "Test stage CMD must run unittest discover",
        )


class TestBDockerignore(unittest.TestCase):
    """Assertions for .dockerignore."""

    def setUp(self):
        raw = (_REPO / ".dockerignore").read_text(encoding="utf-8")
        self.lines = [l.strip() for l in raw.splitlines() if l.strip()]

    def test_b01_required_ignored_entries(self):
        required = [
            ".env",
            "token.json",
            "credentials.json",
            "data/",
            "deploy/*.env",
            ".git",
            "venv/",
            "n8n-workflows/.env.n8n",
        ]
        for item in required:
            self.assertIn(
                item,
                self.lines,
                f".dockerignore must list {item!r}",
            )

    def test_b02_not_ignored_directories(self):
        forbidden = [
            "tests",
            "tests/",
            ".gitignore",
            "deploy",
            "deploy/",
            "n8n-workflows",
            "n8n-workflows/",
        ]
        for item in forbidden:
            self.assertNotIn(
                item,
                self.lines,
                f".dockerignore must not list {item!r}",
            )


class TestCGitignore(unittest.TestCase):
    """Assertions for .gitignore."""

    def setUp(self):
        raw = (_REPO / ".gitignore").read_text(encoding="utf-8")
        self.lines = [l.strip() for l in raw.splitlines() if l.strip()]

    def test_c01_required_ignored_entries(self):
        required = [
            ".env",
            "token.json",
            "credentials.json",
            "deploy/*.env",
            "data/",
            "n8n-workflows/.env.n8n",
        ]
        for item in required:
            self.assertIn(
                item,
                self.lines,
                f".gitignore must contain {item!r}",
            )


class TestDCompose(unittest.TestCase):
    """Assertions for deploy/docker-compose.yml."""

    def setUp(self):
        self.raw = (_REPO / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
        self.stripped = strip_comments(self.raw)
        self.services = parse_compose_services(self.raw)

    def _extract_ports(self, service_block: str) -> list[str]:
        lines = service_block.splitlines()
        in_ports = False
        ports = []
        for l in lines:
            if re.match(r"^ {4}ports:\s*$", l):
                in_ports = True
                continue
            if in_ports:
                m = re.match(r'^ {6}-\s*["\']?([^"\']+)["\']?\s*$', l)
                if m:
                    ports.append(m.group(1).strip())
                else:
                    break
        return ports

    def test_d01_exact_services(self):
        self.assertEqual(
            set(self.services.keys()),
            {"caddy", "backend", "n8n"},
            f"Compose services must be exactly caddy, backend, n8n; got {set(self.services.keys())}",
        )

    def test_d02_port_publishing_rules(self):
        caddy_ports = self._extract_ports(self.services["caddy"])
        self.assertEqual(
            caddy_ports,
            ["80:80", "443:443", "443:443/udp"],
            f"caddy must publish exactly 80:80, 443:443, 443:443/udp; got {caddy_ports}",
        )

        n8n_ports = self._extract_ports(self.services["n8n"])
        self.assertEqual(
            n8n_ports,
            ["127.0.0.1:5678:5678"],
            f"n8n must publish only 127.0.0.1:5678:5678; got {n8n_ports}",
        )

        self.assertNotIn(
            "ports:",
            self.services["backend"],
            "backend must have NO 'ports:' key",
        )
        self.assertRegex(
            self.services["backend"],
            r'expose:\s*\n\s+-\s*["\']?8000["\']?',
            "backend must expose 8000",
        )

    def test_d03_images_and_targets(self):
        self.assertRegex(
            self.services["n8n"],
            r"image:\s*docker\.n8n\.io/n8nio/n8n:2\.47\.7\b",
            "n8n image must be exactly docker.n8n.io/n8nio/n8n:2.47.7",
        )
        self.assertNotIn("latest", self.services["n8n"])
        self.assertRegex(
            self.services["backend"],
            r"target:\s*runtime\b",
            "backend must build target runtime",
        )

    def test_d04_env_files(self):
        for svc, env_f in [("caddy", "caddy.env"), ("backend", "backend.env"), ("n8n", "n8n.env")]:
            self.assertIn(
                env_f,
                self.services[svc],
                f"{svc} must use env_file {env_f}",
            )

    def test_d05_backend_mount_and_token_path(self):
        self.assertIn(
            "../data/backend:/data",
            self.services["backend"],
            "backend must mount ../data/backend:/data",
        )
        self.assertIn(
            "GMAIL_TOKEN_PATH=/data/token.json",
            self.services["backend"],
            "backend must set GMAIL_TOKEN_PATH=/data/token.json",
        )

    def test_d06_n8n_configuration_variables(self):
        for setting in [
            "N8N_BLOCK_ENV_ACCESS_IN_NODE=false",
            "BACKEND_BASE_URL=http://backend:8000",
            "N8N_PROXY_HOPS=1",
            "EXECUTIONS_DATA_PRUNE=true",
        ]:
            self.assertIn(
                setting,
                self.services["n8n"],
                f"n8n must set {setting}",
            )
        self.assertRegex(
            self.services["n8n"],
            r"N8N_CONCURRENCY_PRODUCTION_LIMIT=\d+",
            "n8n must set N8N_CONCURRENCY_PRODUCTION_LIMIT",
        )

    def test_d07_security_options_and_read_only(self):
        for svc in ["caddy", "backend", "n8n"]:
            self.assertTrue(
                "no-new-privileges:true" in self.services[svc] or "no-new-privileges" in self.services[svc],
                f"{svc} must set no-new-privileges",
            )
        self.assertRegex(
            self.services["backend"],
            r"read_only:\s*true",
            "backend must be read_only: true",
        )

    def test_d08_single_lead_net_network(self):
        net_match = re.search(r"^networks:\s*\n\s+([a-zA-Z0-9_\-]+):", self.stripped, re.MULTILINE)
        self.assertIsNotNone(net_match, "top-level networks block not found")
        self.assertEqual(
            net_match.group(1),
            "lead_net",
            f"network must be 'lead_net', got {net_match.group(1)}",
        )

    def test_d09_no_forbidden_keys(self):
        for forbidden in ["privileged", "network_mode", "docker.sock"]:
            self.assertNotIn(
                forbidden,
                self.stripped,
                f"docker-compose.yml must not contain {forbidden!r}",
            )

    def test_d10_log_rotation_present_for_all_services(self):
        for svc in ["caddy", "backend", "n8n"]:
            has_logging = ("logging: *logging" in self.services[svc]) or ("max-size" in self.services[svc])
            self.assertTrue(has_logging, f"{svc} must configure log rotation")

    def test_d11_no_secrets_in_compose(self):
        pattern = r"(?:API_SECRET_KEY|BACKEND_API_KEY)\s*=\s*[^\s$]"
        self.assertIsNone(
            re.search(pattern, self.stripped),
            "No hardcoded API key values permitted in compose file",
        )
        self.assertIsNone(
            re.search(r"(?:sk-or-|xox[abprs]-|hooks\.slack\.com|Bearer\s)", self.stripped),
            "No secret tokens permitted in compose file",
        )


    def test_d12_site_mount_on_caddy_only(self):
        self.assertIn(
            "./site:/srv/site:ro",
            self.services["caddy"],
            "caddy service must mount ./site:/srv/site:ro",
        )
        for svc in ["backend", "n8n"]:
            self.assertNotIn(
                "/srv/site",
                self.services[svc],
                f"{svc} must not mount /srv/site",
            )
            self.assertNotIn(
                "./site",
                self.services[svc],
                f"{svc} must not mount ./site",
            )


class TestECaddyfile(unittest.TestCase):
    """Assertions for deploy/Caddyfile."""

    def setUp(self):
        self.raw = (_REPO / "deploy" / "Caddyfile").read_text(encoding="utf-8")
        self.stripped = strip_comments(self.raw)

    def test_e01_structural_markers(self):
        self.assertIn("admin off", self.stripped)
        self.assertIn("{$APP_DOMAIN}", self.stripped)
        self.assertRegex(self.stripped, r"request_body\s*\{[^}]*max_size")
        self.assertIn("Strict-Transport-Security", self.stripped)
        self.assertIn('respond "Not Found" 404', self.stripped)

    def test_e02_path_matchers_exact_tokens(self):
        path_lines = re.findall(r"@\w+\s+path\s+([^\n]+)", self.stripped)
        tokens = []
        for pl in path_lines:
            tokens.extend(pl.strip().split())
        expected = [
            "/webhook/lead-intake",
            "/health",
            "/email/auth",
            "/email/auth/callback",
            "/lead-form",
            "/lead-form.js",
            "/",
        ]
        self.assertEqual(
            sorted(tokens),
            sorted(expected),
            f"Path matchers must contain exactly {expected}, got {tokens}",
        )

    def test_e03_no_internal_routes_exposed(self):
        banned = re.findall(r"/(?:leads|ai|sms|crm|docs|openapi|redoc)\b", self.stripped)
        self.assertEqual(
            banned,
            [],
            f"Banned path tokens found in Caddyfile: {banned}",
        )

    def test_e04_reverse_proxy_targets_and_counts(self):
        rp_n8n = re.findall(r"reverse_proxy\s+n8n:5678", self.stripped)
        self.assertEqual(len(rp_n8n), 1, "reverse_proxy n8n:5678 must appear exactly once")

        intake_match = re.search(r"handle\s+@intake\s*\{([\s\S]*?)\n\t\}", self.stripped)
        self.assertIsNotNone(intake_match, "handle @intake block not found")
        self.assertIn(
            "reverse_proxy n8n:5678",
            intake_match.group(1),
            "reverse_proxy n8n:5678 must be inside the /webhook/lead-intake handler",
        )

        rp_backend = re.findall(r"reverse_proxy\s+backend:8000", self.stripped)
        self.assertEqual(len(rp_backend), 2, "reverse_proxy backend:8000 must appear exactly twice")

    def test_e05_rate_limits_inside_route_blocks(self):
        rate_limits = list(re.finditer(r"rate_limit\s*\{", self.stripped))
        self.assertEqual(len(rate_limits), 3, "Expected 3 rate_limit blocks")
        for m in rate_limits:
            prefix = self.stripped[:m.start()]
            last_route = prefix.rfind("route {")
            self.assertGreater(last_route, -1, "rate_limit must be inside a route block")
            between = prefix[last_route:]
            opens = between.count("{")
            closes = between.count("}")
            self.assertGreater(opens, closes, "rate_limit must be inside open route block")

    def test_e06_intake_handler_details(self):
        intake_match = re.search(r"handle\s+@intake\s*\{([\s\S]*?)\n\t\}", self.stripped)
        self.assertIsNotNone(intake_match)
        intake_body = intake_match.group(1)
        self.assertRegex(
            intake_body,
            r"key\s+\{remote_host\}",
            "intake rate limit must have a per-IP zone keyed {remote_host}",
        )
        static_keys = re.findall(r"key\s+static", intake_body)
        self.assertEqual(
            len(static_keys),
            2,
            f"intake rate limit must have two zones with 'key static', found {len(static_keys)}",
        )
        self.assertRegex(
            intake_body,
            r"respond\s+@not_post\s+[\"']Method Not Allowed[\"']\s+405",
            "intake handler must answer non-POST with 405",
        )


    def test_e07_lead_form_page_handler(self):
        page_match = re.search(r"handle\s+@form_page\s*\{([\s\S]*?)\n\t\}", self.stripped)
        self.assertIsNotNone(page_match, "handle @form_page block not found")
        body = page_match.group(1)
        self.assertIn("templates", body, "Page handler must contain 'templates'")
        self.assertRegex(
            body,
            r'respond\s+@not_get_head\s+["\']Method Not Allowed["\']\s+405',
            "Page handler must answer non-GET/HEAD with 405",
        )
        self.assertIn('Cache-Control "no-store"', body, "Page handler must set Cache-Control 'no-store'")

        idx_root = body.find("root * /srv/site")
        idx_rewrite = body.find("rewrite * /lead-form.html")
        idx_header = body.find("header {")
        idx_templates = body.find("templates")
        idx_fileserver = body.find("file_server")
        self.assertTrue(
            -1 < idx_root < idx_rewrite < idx_header < idx_templates < idx_fileserver,
            f"Expected order root < rewrite < header < templates < file_server, got: "
            f"root={idx_root}, rewrite={idx_rewrite}, header={idx_header}, templates={idx_templates}, file_server={idx_fileserver}",
        )

    def test_e08_lead_form_script_handler(self):
        js_match = re.search(r"handle\s+@form_js\s*\{([\s\S]*?)\n\t\}", self.stripped)
        self.assertIsNotNone(js_match, "handle @form_js block not found")
        body = js_match.group(1)
        self.assertIn("file_server", body, "Script handler must contain 'file_server'")
        self.assertNotIn("templates", body, "Script handler must NOT contain 'templates'")
        self.assertRegex(
            body,
            r'respond\s+@not_get_head\s+["\']Method Not Allowed["\']\s+405',
            "Script handler must answer non-GET/HEAD with 405",
        )
        self.assertIn('Cache-Control "public, max-age=300"', body, "Script handler must set Cache-Control 'public, max-age=300'")

        idx_root = body.find("root * /srv/site")
        idx_header = body.find("header {")
        idx_fileserver = body.find("file_server")
        self.assertTrue(
            -1 < idx_root < idx_header < idx_fileserver,
            f"Expected order root < header < file_server, got: root={idx_root}, header={idx_header}, file_server={idx_fileserver}",
        )

    def test_e09_csp_headers_and_security(self):
        for handler_name in ["@form_page", "@form_js"]:
            m = re.search(rf"handle\s+{handler_name}\s*\{{([\s\S]*?)\n\t\}}", self.stripped)
            self.assertIsNotNone(m, f"handle {handler_name} block not found")
            body = m.group(1)
            self.assertIn("Content-Security-Policy", body)
            self.assertIn("script-src 'self' https://challenges.cloudflare.com", body)
            self.assertIn("frame-ancestors 'none'", body)
            self.assertNotIn("unsafe-eval", body)
            self.assertNotRegex(
                body,
                r"script-src[^;\"]*'unsafe-inline'",
                f"{handler_name} must not allow 'unsafe-inline' in script-src",
            )
            self.assertIn("style-src 'self' 'unsafe-inline'", body)

    def test_e10_root_srv_site_and_no_other_file_server_or_templates(self):
        self.assertIn("root * /srv/site", self.stripped)
        fs_matches = re.findall(r"\bfile_server\b", self.stripped)
        self.assertEqual(len(fs_matches), 3, f"file_server must appear three times, got {len(fs_matches)}")
        tmpl_matches = re.findall(r"\btemplates\b", self.stripped)
        self.assertEqual(len(tmpl_matches), 1, f"templates must appear exactly once, got {len(tmpl_matches)}")

    def test_e11_showcase_home_handler(self):
        home_match = re.search(r"handle\s+@home\s*\{([\s\S]*?)\n\t\}", self.stripped)
        self.assertIsNotNone(home_match, "handle @home block not found")
        body = home_match.group(1)

        self.assertIn('respond @not_get_head "Method Not Allowed" 405', body)
        self.assertIn('Cache-Control "public, max-age=300"', body)

        expected_csp = (
            "default-src 'none'; style-src 'unsafe-inline'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
        csp_match = re.search(r'Content-Security-Policy\s+"([^"]+)"', body)
        self.assertIsNotNone(csp_match, "handle @home must set Content-Security-Policy")
        csp_value = csp_match.group(1)
        self.assertEqual(csp_value, expected_csp)

        for forbidden in ("script-src", "unsafe-eval", "img-src", "http:", "https:"):
            self.assertNotIn(forbidden, csp_value, f"Home CSP must not contain {forbidden!r}")

        for forbidden in ("templates", "rate_limit", "reverse_proxy"):
            self.assertNotIn(forbidden, body, f"handle @home must not contain {forbidden!r}")

        idx_root = body.find("root * /srv/site")
        idx_rewrite = body.find("rewrite * /index.html")
        idx_header = body.find("header {")
        idx_fileserver = body.find("file_server")
        self.assertTrue(
            -1 < idx_root < idx_rewrite < idx_header < idx_fileserver,
            f"Expected order root < rewrite < header < file_server, got: "
            f"root={idx_root}, rewrite={idx_rewrite}, header={idx_header}, file_server={idx_fileserver}",
        )

        rewrites = re.findall(r"rewrite\s+\*\s+/index\.html(?:\s|$)", self.stripped)
        self.assertEqual(
            len(rewrites),
            1,
            f"Expected exactly one 'rewrite * /index.html', got {len(rewrites)}",
        )

        home_matchers = re.findall(r"(?m)^\s*@home\s+path\s+/\s*$", self.stripped)
        self.assertEqual(
            len(home_matchers),
            1,
            f"Expected '@home path /' exactly once, got {len(home_matchers)}",
        )


class TestFCaddyDockerfile(unittest.TestCase):
    """Assertions for deploy/caddy/Dockerfile."""

    def setUp(self):
        self.raw = (_REPO / "deploy" / "caddy" / "Dockerfile").read_text(encoding="utf-8")

    def test_f01_uses_xcaddy_with_ratelimit_and_caddy_2(self):
        self.assertIn(
            "github.com/mholt/caddy-ratelimit",
            self.raw,
            "deploy/caddy/Dockerfile must build with github.com/mholt/caddy-ratelimit",
        )
        from_lines = re.findall(r"(?m)^FROM\s+([^\s]+)", self.raw)
        self.assertGreaterEqual(len(from_lines), 2)
        self.assertEqual(
            from_lines[-1],
            "caddy:2",
            f"last FROM in deploy/caddy/Dockerfile must be caddy:2, got {from_lines[-1]!r}",
        )


class TestGEnvExamples(unittest.TestCase):
    """Assertions for deploy/*.env.example files."""

    def setUp(self):
        self.backend = parse_env_file(_REPO / "deploy" / "backend.env.example")
        self.n8n = parse_env_file(_REPO / "deploy" / "n8n.env.example")
        self.caddy = parse_env_file(_REPO / "deploy" / "caddy.env.example")

    def test_g01_secret_like_keys_empty(self):
        secret_keys = {
            "API_SECRET_KEY",
            "BACKEND_API_KEY",
            "OPENROUTER_API_KEY",
            "GMAIL_CLIENT_SECRET",
            "TWILIO_AUTH_TOKEN",
            "SUPABASE_KEY",
            "HUBSPOT_ACCESS_TOKEN",
            "SLACK_LEADS_WEBHOOK_URL",
            "SLACK_ALERTS_WEBHOOK_URL",
            "WEBHOOK_SHARED_SECRET",
            "TURNSTILE_SECRET_KEY",
        }
        for name, env_dict in [("backend", self.backend), ("n8n", self.n8n), ("caddy", self.caddy)]:
            for k in secret_keys:
                if k in env_dict:
                    self.assertEqual(
                        env_dict[k],
                        "",
                        f"In {name}.env.example, {k} must have an empty value; got {env_dict[k]!r}",
                    )

    def test_g02_dry_run_flags_are_true(self):
        self.assertEqual(self.backend.get("GMAIL_DRY_RUN"), "true")
        self.assertEqual(self.backend.get("TWILIO_DRY_RUN"), "true")
        self.assertEqual(self.backend.get("HUBSPOT_DRY_RUN"), "true")

    def test_g03_urls_and_endpoints(self):
        gmail_uri = self.backend.get("GMAIL_REDIRECT_URI", "")
        self.assertTrue(
            gmail_uri.startswith("https://") and gmail_uri.endswith("/email/auth/callback"),
            f"GMAIL_REDIRECT_URI must start with https:// and end with /email/auth/callback, got {gmail_uri!r}",
        )

        n8n_url = self.n8n.get("WEBHOOK_URL", "")
        self.assertTrue(
            n8n_url.startswith("https://") and n8n_url.endswith("/"),
            f"WEBHOOK_URL must start with https:// and end with '/', got {n8n_url!r}",
        )

        self.assertIn("APP_DOMAIN", self.caddy)
        self.assertIn("ACME_EMAIL", self.caddy)


    def test_g04_turnstile_sitekey_defined_in_caddy(self):
        self.assertIn("TURNSTILE_SITEKEY", self.caddy)
        self.assertEqual(self.caddy["TURNSTILE_SITEKEY"], "")


class TestHDriftGuard(unittest.TestCase):
    """Drift guard: ensures environment variables in code are documented in backend.env.example."""

    def test_h01_backend_env_example_drift_guard(self):
        files = [_REPO / "main.py"] + list((_REPO / "services").glob("*.py"))
        pattern = re.compile(
            r"(?:os\.getenv|os\.environ\.get|_require_env)\s*\(\s*[\"\']([A-Z0-9_]+)[\"\']",
            re.MULTILINE | re.DOTALL,
        )

        discovered = set()
        for f in files:
            content = f.read_text(encoding="utf-8")
            discovered.update(pattern.findall(content))

        self.assertGreaterEqual(
            len(discovered),
            20,
            f"Drift guard expected at least 20 distinct environment variables, found {len(discovered)}",
        )
        self.assertIn(
            "APP_DESCRIPTION",
            discovered,
            "Drift guard must discover APP_DESCRIPTION across line breaks",
        )

        example_text = (_REPO / "deploy" / "backend.env.example").read_text(encoding="utf-8")
        example_keys = set(re.findall(r"^([A-Z0-9_]+)=", example_text, re.MULTILINE))

        exemptions = {"GMAIL_TOKEN_PATH", "HOME"}
        unmatched = (discovered - exemptions) - example_keys

        self.assertEqual(
            unmatched,
            set(),
            f"Variables found in source code but missing from deploy/backend.env.example: {sorted(unmatched)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
