# tests/test_showcase_site.py
# Standard-library only: unittest, re, pathlib, html.parser
# Run with: python -m unittest discover -s tests -t . -v

from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

_REPO = Path(__file__).resolve().parent.parent
_HTML_PATH = _REPO / "deploy" / "site" / "index.html"
_MAX_BYTES = 35000

_FORBIDDEN_TAGS = (
    "script", "link", "img", "image", "iframe", "form",
    "input", "foreignobject", "use",
)
_REQUIRED_CHANNELS = (
    "Lead intake form with CAPTCHA",
    "AI scoring",
    "Storage (database)",
    "Slack summary",
    "Follow-up email",
    "Follow-up SMS",
    "CRM sync",
)
_REQUIRED_TECH = (
    "FastAPI", "n8n", "Caddy", "Cloudflare Turnstile",
    "OpenRouter", "Supabase", "Docker",
)
_BANNED_WORDS = (
    "enterprise",
    "production-ready",
    "robust",
    "scalable",
    "zero downtime",
    "bulletproof",
    "secure",
    "ai-powered",
    "under 60 seconds",
    "honest limits",
)
_PLACEHOLDER_TOKENS = ("TODO", "FIXME", "<PASTE", "your-name")


class _ShowcaseParser(HTMLParser):
    """Parse deploy/site/index.html for structure, CSP, and visible text."""

    def __init__(self):
        super().__init__()
        self.tag_counts = {}
        self.html_attrs = {}
        self.metas = []
        self.titles = []        # <title> texts
        self.links = []         # list of attrs dicts for <a>
        self.all_attrs = []     # (tag, attr_name, attr_value)
        self.ids = set()        # all id= values found
        # SVG tracking
        self.svgs = []          # list of attrs_dict per <svg>
        self._svg_depth = 0
        self._svg_has_title = False
        self._svg_has_desc = False
        self._svg_checks = []   # list of (has_title, has_desc) per closed svg
        # text tracking
        self._in_title = False
        self._in_style = False
        self._title_chunks = []
        self._visible_chunks = []

    def handle_starttag(self, tag, attrs):
        tag_l = tag.lower()
        attrs_dict = {k.lower(): ("" if v is None else v) for k, v in attrs}
        self.tag_counts[tag_l] = self.tag_counts.get(tag_l, 0) + 1
        for k, v in attrs:
            self.all_attrs.append((tag_l, k, v))
        if "id" in attrs_dict:
            self.ids.add(attrs_dict["id"])

        if tag_l == "html":
            self.html_attrs = attrs_dict
        elif tag_l == "meta":
            self.metas.append(attrs_dict)
        elif tag_l == "title":
            if self._svg_depth == 0:
                self._in_title = True
                self._title_chunks = []
            else:
                # <title> inside SVG
                self._svg_has_title = True
        elif tag_l == "desc":
            if self._svg_depth > 0:
                self._svg_has_desc = True
        elif tag_l == "style":
            self._in_style = True
        elif tag_l == "a":
            self.links.append(attrs_dict)
        elif tag_l == "svg":
            if self._svg_depth == 0:
                self.svgs.append(attrs_dict)
                self._svg_has_title = False
                self._svg_has_desc = False
            self._svg_depth += 1

    def handle_endtag(self, tag):
        tag_l = tag.lower()
        if tag_l == "title" and self._in_title:
            self.titles.append("".join(self._title_chunks).strip())
            self._in_title = False
            self._title_chunks = []
        elif tag_l == "style":
            self._in_style = False
        elif tag_l == "svg" and self._svg_depth > 0:
            self._svg_depth -= 1
            if self._svg_depth == 0:
                self._svg_checks.append(
                    (self._svg_has_title, self._svg_has_desc)
                )

    def handle_data(self, data):
        if self._in_title:
            self._title_chunks.append(data)
        if not self._in_style:
            stripped = data.strip()
            if stripped:
                self._visible_chunks.append(stripped)

    @property
    def visible_text(self):
        return " ".join(self._visible_chunks)


def _load():
    raw_bytes = _HTML_PATH.read_bytes()
    raw_text = raw_bytes.decode("ascii")
    parser = _ShowcaseParser()
    parser.feed(raw_text)
    parser.close()
    return raw_text, parser


# ---------------------------------------------------------------------------
# Group A: File hygiene
# ---------------------------------------------------------------------------
class TestAGroupFileHygiene(unittest.TestCase):

    def test_a01_exists_and_under_limit(self):
        self.assertTrue(_HTML_PATH.is_file(), f"Missing: {_HTML_PATH}")
        raw = _HTML_PATH.read_bytes()
        self.assertLess(len(raw), _MAX_BYTES,
                        f"File size {len(raw)} >= {_MAX_BYTES} bytes")

    def test_a02_pure_ascii_no_bom(self):
        raw = _HTML_PATH.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "BOM found")
        raw.decode("ascii")  # raises UnicodeDecodeError if non-ASCII

    def test_a03_lf_only_single_trailing(self):
        raw = _HTML_PATH.read_bytes()
        self.assertNotIn(b"\r", raw, "CR found")
        self.assertTrue(raw.endswith(b"\n"), "Must end with LF")
        self.assertFalse(raw.endswith(b"\n\n"), "Double trailing LF")


# ---------------------------------------------------------------------------
# Group B: HTML structure and CSP
# ---------------------------------------------------------------------------
class TestBGroupHtmlStructure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.p = _load()

    def test_b01_document_metadata(self):
        self.assertEqual(self.p.html_attrs.get("lang"), "en")
        self.assertEqual(len(self.p.titles), 1, "Expected exactly one <title>")
        self.assertTrue(self.p.titles[0], "<title> must be non-empty")
        vp = [m for m in self.p.metas if m.get("name", "").lower() == "viewport"]
        self.assertEqual(len(vp), 1, "Expected one <meta name=viewport>")
        self.assertTrue(vp[0].get("content"))
        desc = [m for m in self.p.metas if m.get("name", "").lower() == "description"]
        self.assertEqual(len(desc), 1, "Expected one <meta name=description>")
        self.assertTrue(desc[0].get("content"))

    def test_b02_one_h1_one_style_landmarks(self):
        self.assertEqual(self.p.tag_counts.get("h1", 0), 1)
        self.assertEqual(self.p.tag_counts.get("style", 0), 1)
        for lm in ("header", "main", "footer", "section"):
            with self.subTest(landmark=lm):
                self.assertGreaterEqual(self.p.tag_counts.get(lm, 0), 1,
                                        f"No <{lm}> found")

    def test_b03_forbidden_tags_absent(self):
        for tag in _FORBIDDEN_TAGS:
            with self.subTest(tag=tag):
                self.assertEqual(self.p.tag_counts.get(tag, 0), 0,
                                 f"Forbidden <{tag}> found")

    def test_b04_no_on_attributes(self):
        for tag, name, _ in self.p.all_attrs:
            self.assertFalse(name.lower().startswith("on"),
                             f"<{tag}> has event attr {name!r}")

    def test_b05_links_relative_and_lead_form_twice(self):
        self.assertGreater(len(self.p.links), 0, "No links found")
        lf_count = 0
        for a in self.p.links:
            href = a.get("href", "")
            with self.subTest(href=href):
                self.assertTrue(href.startswith("/") or href.startswith("#"),
                                f"Non-relative href {href!r}")
            if href == "/lead-form":
                lf_count += 1
        self.assertGreaterEqual(lf_count, 2,
                                f"Need >=2 /lead-form links, got {lf_count}")

    def test_b06_hash_links_have_matching_ids(self):
        for a in self.p.links:
            href = a.get("href", "")
            if href.startswith("#"):
                target = href[1:]
                self.assertIn(target, self.p.ids,
                              f"Hash link {href!r} has no matching id in document")

    def test_b07_exactly_one_svg_with_role_aria_no_xmlns(self):
        self.assertEqual(self.p.tag_counts.get("svg", 0), 1,
                         "Expected exactly one <svg>")
        svg_attrs = self.p.svgs[0]
        self.assertEqual(svg_attrs.get("role"), "img",
                         "<svg> must have role='img'")
        self.assertTrue(svg_attrs.get("aria-label", "").strip(),
                        "<svg> must have non-empty aria-label")
        self.assertNotIn("xmlns", svg_attrs,
                         "<svg> must NOT have xmlns attribute")

    def test_b08_svg_contains_title_and_desc(self):
        self.assertEqual(len(self.p._svg_checks), 1,
                         "Expected SVG closing check entry")
        has_title, has_desc = self.p._svg_checks[0]
        self.assertTrue(has_title, "<svg> must contain a <title> element")
        self.assertTrue(has_desc, "<svg> must contain a <desc> element")


# ---------------------------------------------------------------------------
# Group C: Content guards
# ---------------------------------------------------------------------------
class TestCGroupContentAndGuards(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw, cls.p = _load()
        cls.vis = cls.p.visible_text

    def test_c01_no_absolute_urls(self):
        m = re.search(r"https?://", self.raw, re.IGNORECASE)
        self.assertIsNone(m, f"http/https URL found")

    def test_c02_required_keywords_in_visible_text(self):
        lower = self.vis.lower()
        for kw in ("dry-run", "fake", "live"):
            with self.subTest(kw=kw):
                self.assertIn(kw, lower)

    def test_c03_all_channels_in_visible_text(self):
        for ch in _REQUIRED_CHANNELS:
            with self.subTest(ch=ch):
                self.assertIn(ch, self.vis)

    def test_c04_all_tech_names_in_visible_text(self):
        for tech in _REQUIRED_TECH:
            with self.subTest(tech=tech):
                self.assertIn(tech, self.vis)

    def test_c05_slack_mockup_markers(self):
        for marker in ("HOT lead", "Illustrative example", "nothing was sent"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.vis)

    def test_c06_live_and_dry_run_pills_count(self):
        # Count occurrences of exact pill text in visible text
        live_count = self.vis.count("LIVE")
        dry_count  = self.vis.count("DRY-RUN")
        self.assertGreaterEqual(live_count, 3,
                                f"Expected >=3 LIVE in visible text, got {live_count}")
        self.assertGreaterEqual(dry_count, 3,
                                f"Expected >=3 DRY-RUN in visible text, got {dry_count}")

    def test_c07_no_three_digit_sequences_except_budget(self):
        # Remove the allowed exception before checking
        sanitised = self.vis.replace("$450k-$550k", "")
        matches = re.findall(r"\d{3,}", sanitised)
        self.assertEqual(matches, [],
                         f"3+ digit sequences found (after removing exception): {matches}")

    def test_c08_no_banned_words(self):
        lower = self.raw.lower()
        for word in _BANNED_WORDS:
            with self.subTest(word=word):
                self.assertNotIn(word, lower,
                                 f"Banned word {word!r} found")

    def test_c09_no_placeholder_tokens(self):
        lower = self.raw.lower()
        for token in _PLACEHOLDER_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token.lower(), lower,
                                 f"Placeholder {token!r} found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
