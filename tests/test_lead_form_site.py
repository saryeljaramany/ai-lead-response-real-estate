# tests/test_lead_form_site.py
# Standard-library only: unittest, re, pathlib, html.parser, json, subprocess, shutil, tempfile
# Run with: python -m unittest discover -s tests -t . -v

import json
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

_REPO = Path(__file__).resolve().parent.parent
_HTML_PATH = _REPO / "deploy" / "site" / "lead-form.html"
_JS_PATH = _REPO / "deploy" / "site" / "lead-form.js"

for label, p in [("deploy/site/lead-form.html", _HTML_PATH), ("deploy/site/lead-form.js", _JS_PATH)]:
    if not p.is_file():
        raise FileNotFoundError(f"Required site file missing: {label} (resolved to {p})")

_RAW_HTML_BYTES = _HTML_PATH.read_bytes()
_RAW_JS_BYTES = _JS_PATH.read_bytes()

_RAW_HTML = _RAW_HTML_BYTES.decode("ascii")
_RAW_JS = _RAW_JS_BYTES.decode("ascii")


class _LeadFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.scripts = []  # list of (attrs_dict, data_content)
        self.current_script_attrs = None
        self.current_script_data = []
        self.forms = []  # list of attrs_dict
        self.fields = []  # list of (tag, attrs_dict)
        self.labels = []  # list of attrs_dict
        self.options = []  # list of (select_name, value)
        self.current_select = None
        self.elements_by_id = {}
        self.all_attrs = []  # list of (tag, attr_name, attr_value)
        self.cf_turnstile_elements = []  # list of attrs_dict

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.tags.append((tag, attrs_dict))
        for k, v in attrs:
            self.all_attrs.append((tag, k, v))

        if "id" in attrs_dict:
            self.elements_by_id[attrs_dict["id"]] = (tag, attrs_dict)

        if tag == "script":
            self.current_script_attrs = attrs_dict
            self.current_script_data = []

        if tag == "form":
            self.forms.append(attrs_dict)

        if tag in ("input", "select", "textarea"):
            if "name" in attrs_dict:
                self.fields.append((tag, attrs_dict))
            if tag == "select":
                self.current_select = attrs_dict.get("name")

        if tag == "label":
            self.labels.append(attrs_dict)

        if tag == "option" and self.current_select:
            self.options.append((self.current_select, attrs_dict.get("value", "")))

        classes = attrs_dict.get("class", "").split()
        if "cf-turnstile" in classes:
            self.cf_turnstile_elements.append(attrs_dict)

    def handle_endtag(self, tag):
        if tag == "script" and self.current_script_attrs is not None:
            data = "".join(self.current_script_data).strip()
            self.scripts.append((self.current_script_attrs, data))
            self.current_script_attrs = None
            self.current_script_data = []

        if tag == "select":
            self.current_select = None

    def handle_data(self, data):
        if self.current_script_attrs is not None:
            self.current_script_data.append(data)


# Group A: File encodings and line endings
class TestAGroupFileEncodings(unittest.TestCase):

    def test_a01_both_files_exist_and_pure_ascii(self):
        for name, b in [("lead-form.html", _RAW_HTML_BYTES), ("lead-form.js", _RAW_JS_BYTES)]:
            with self.subTest(file=name):
                b.decode("ascii")

    def test_a02_no_cr_in_either_file(self):
        for name, b in [("lead-form.html", _RAW_HTML_BYTES), ("lead-form.js", _RAW_JS_BYTES)]:
            with self.subTest(file=name):
                self.assertNotIn(b"\r", b, f"{name} contains carriage return (CR)")

    def test_a03_no_utf8_bom(self):
        for name, b in [("lead-form.html", _RAW_HTML_BYTES), ("lead-form.js", _RAW_JS_BYTES)]:
            with self.subTest(file=name):
                self.assertFalse(b.startswith(b"\xef\xbb\xbf"), f"{name} has UTF-8 BOM")

    def test_a04_single_trailing_lf(self):
        for name, b in [("lead-form.html", _RAW_HTML_BYTES), ("lead-form.js", _RAW_JS_BYTES)]:
            with self.subTest(file=name):
                self.assertTrue(b.endswith(b"\n"), f"{name} must end with LF")
                self.assertFalse(b.endswith(b"\n\n"), f"{name} must not end with multiple LFs")


# Group B: HTML Parsing and Structure
class TestBGroupHtmlStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = _LeadFormParser()
        cls.parser.feed(_RAW_HTML)

    def test_b01_exactly_two_script_tags_both_with_src_no_inline_body(self):
        self.assertEqual(len(self.parser.scripts), 2, "Expected exactly 2 script tags")
        srcs = [attrs.get("src") for attrs, _ in self.parser.scripts]
        expected_srcs = [
            "https://challenges.cloudflare.com/turnstile/v0/api.js",
            "/lead-form.js",
        ]
        self.assertEqual(srcs, expected_srcs, f"Expected scripts {expected_srcs}, got {srcs}")
        for attrs, body in self.parser.scripts:
            self.assertEqual(body, "", f"Script with src {attrs.get('src')!r} must have no inline body")

    def test_b02_no_inline_event_handler_attributes(self):
        for tag, attr_name, _ in self.parser.all_attrs:
            self.assertFalse(
                attr_name.lower().startswith("on"),
                f"Tag <{tag}> has inline event handler attribute {attr_name!r}"
            )

    def test_b03_only_one_absolute_url_in_html(self):
        urls = re.findall(r"https?://[^\s\"'>]+", _RAW_HTML)
        expected = ["https://challenges.cloudflare.com/turnstile/v0/api.js"]
        self.assertEqual(urls, expected, f"Unexpected URLs in HTML: {urls}")

    def test_b04_form_has_id_lead_form(self):
        self.assertEqual(len(self.parser.forms), 1, "Expected exactly one <form>")
        form = self.parser.forms[0]
        self.assertEqual(form.get("id"), "lead-form")

    def test_b05_form_field_names_exact_set(self):
        field_names = {attrs["name"] for _, attrs in self.parser.fields}
        expected = {
            "name", "email", "phone", "message",
            "property_type", "preferred_location", "budget", "timeline"
        }
        self.assertEqual(field_names, expected)

    def test_b06_every_field_has_matching_label(self):
        label_fors = {attrs.get("for") for attrs in self.parser.labels if "for" in attrs}
        for _, attrs in self.parser.fields:
            field_id = attrs.get("id")
            field_name = attrs.get("name")
            self.assertEqual(field_id, field_name, f"Field {field_name!r} id must match name")
            self.assertIn(
                field_id, label_fors,
                f"Field {field_id!r} has no matching <label for={field_id!r}>"
            )

    def test_b07_maxlength_values_match_n8n_limits(self):
        limits = {
            "name": "200",
            "email": "254",
            "phone": "50",
            "message": "5000",
            "preferred_location": "200",
            "budget": "200",
            "timeline": "200",
        }
        fields_by_name = {attrs["name"]: attrs for _, attrs in self.parser.fields}
        for field, max_val in limits.items():
            self.assertIn(field, fields_by_name, f"Field {field!r} not found")
            self.assertEqual(
                fields_by_name[field].get("maxlength"), max_val,
                f"Field {field!r} maxlength must be {max_val}"
            )

        for select_name, opt_val in self.parser.options:
            if select_name == "property_type":
                self.assertLessEqual(
                    len(opt_val), 200,
                    f"Option value {opt_val!r} in property_type exceeds 200 characters"
                )

    def test_b08_field_input_types(self):
        fields_by_name = {attrs["name"]: attrs for _, attrs in self.parser.fields}
        self.assertEqual(fields_by_name["email"].get("type"), "email")
        self.assertEqual(fields_by_name["phone"].get("type"), "tel")

    def test_b09_required_attribute_on_exact_fields(self):
        fields_by_name = {attrs["name"]: attrs for _, attrs in self.parser.fields}
        required_fields = {name for name, attrs in fields_by_name.items() if "required" in attrs}
        expected_required = {"name", "email", "phone", "message"}
        self.assertEqual(required_fields, expected_required)

    def test_b10_turnstile_widget_element(self):
        self.assertEqual(
            len(self.parser.cf_turnstile_elements), 1,
            "Expected exactly one element with class 'cf-turnstile'"
        )
        elem = self.parser.cf_turnstile_elements[0]
        self.assertEqual(
            elem.get("data-sitekey"),
            '{{env "TURNSTILE_SITEKEY"}}',
            "data-sitekey must match literal '{{env \"TURNSTILE_SITEKEY\"}}'"
        )

    def test_b11_form_status_element_has_role_status(self):
        self.assertIn("form-status", self.parser.elements_by_id)
        tag, attrs = self.parser.elements_by_id["form-status"]
        self.assertEqual(attrs.get("role"), "status")


# Group C: Privacy and Secret Guards
class TestCGroupPrivacyAndSecretGuards(unittest.TestCase):

    def test_c01_no_real_turnstile_sitekey_committed(self):
        pattern = re.compile(r"0x4[A-Za-z0-9_-]{10,}")
        self.assertIsNone(pattern.search(_RAW_HTML), "Real Turnstile sitekey found in lead-form.html")
        self.assertIsNone(pattern.search(_RAW_JS), "Real Turnstile sitekey found in lead-form.js")

    def test_c02_no_secret_in_either_site_file(self):
        pattern = re.compile(r"secret", re.IGNORECASE)
        self.assertIsNone(pattern.search(_RAW_HTML), "Word 'secret' found in lead-form.html")
        self.assertIsNone(pattern.search(_RAW_JS), "Word 'secret' found in lead-form.js")


# Group D: JavaScript Logic and Safeguards
class TestDGroupJsLogic(unittest.TestCase):

    def test_d01_required_tokens_present(self):
        required = [
            "/webhook/lead-intake",
            "captcha_token",
            "cf-turnstile-response",
            "textContent",
            "AbortController",
            "20000",
        ]
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, _RAW_JS, f"lead-form.js must contain {token!r}")

    def test_d02_exact_nine_fixed_messages_present_verbatim(self):
        expected_messages = [
            "Please complete the verification.",
            "Thank you! Your request was received. An agent will contact you shortly.",
            "Please check the form: some required details are missing or invalid.",
            "The verification failed. Please complete it again and resubmit.",
            "Your message is too long.",
            "Too many requests. Please wait a minute and try again.",
            "The service is temporarily unavailable. Please try again later.",
            "Could not reach the server. Check your connection and try again.",
            "Something went wrong. Please try again later.",
        ]
        for msg in expected_messages:
            with self.subTest(msg=msg):
                self.assertIn(msg, _RAW_JS, f"lead-form.js must contain message {msg!r}")

    def test_d03_forbidden_substrings_absent(self):
        forbidden = [
            "innerHTML",
            "outerHTML",
            "insertAdjacentHTML",
            "eval",
            "new Function",
            "document.write",
            "console.",
            "localStorage",
            "sessionStorage",
            "cookie",
            "secret",
        ]
        js_lower = _RAW_JS.lower()
        for token in forbidden:
            with self.subTest(forbidden=token):
                if token == "secret":
                    self.assertNotIn("secret", js_lower, "Word 'secret' must not appear in lead-form.js")
                else:
                    self.assertNotIn(token, _RAW_JS, f"Forbidden substring {token!r} found in lead-form.js")

        self.assertIsNone(
            re.search(r"https?://", _RAW_JS, re.IGNORECASE),
            "lead-form.js must not contain any URL starting with http/https"
        )


# Group E: Syntax check via Node CLI
class TestEGroupNodeCheck(unittest.TestCase):

    def test_e01_node_syntax_check(self):
        node_bin = shutil.which("node")
        if not node_bin:
            self.skipTest("node CLI not found -- skipping syntax check")

        wrapped = "(function(){\n" + _RAW_JS + "\n});\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(wrapped)
            tmp_path = f.name
        try:
            res = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode, 0,
                f"Node syntax check failed: {res.stderr.strip()}"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
