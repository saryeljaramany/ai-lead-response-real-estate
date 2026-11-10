"""
tests/test_calendly_service.py

Unit tests for Phase 8 Calendly booking-link layer.
Enforces all rules, pre-fill formatting, UTM handling, URL stripping,
deterministic copy injection, and SMS length planning.

Pure ASCII source: all non-ASCII characters are written as \\uXXXX escapes.
"""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from models import LeadInput
from services.calendly_service import (
    LINK_TOKEN,
    SMS_MAX_CHARS,
    SMS_OPT_OUT,
    SmsPlan,
    _EMAIL_CTA_FALLBACK,
    _normalize_base_link,
    build_booking_link,
    finish_sms,
    get_base_link,
    inject_link_into_email_body,
    plan_sms,
    strip_urls,
)

BASE = "https://calendly.com/real-estate-team/property-viewing"


def _make_lead(
    name: str = "Sarah Mitchell",
    email: str = "sarah.mitchell@example.com",
    phone: str = "+14155550192",
    message: str = "Looking for a 3-bedroom house.",
) -> LeadInput:
    """Create a sample LeadInput instance with standard defaults."""
    return LeadInput(name=name, email=email, phone=phone, message=message)


# ---------------------------------------------------------------------------
# A. Import safety
# ---------------------------------------------------------------------------


class TestImportSafety(unittest.TestCase):
    """Verify that importing or reloading the module requires no environment."""

    def test_import_with_empty_environment(self):
        """Reloading the module in an empty environment must not raise."""
        with patch.dict(os.environ, {}, clear=True):
            import services.calendly_service as cs

            importlib.reload(cs)

    def test_source_file_decodes_as_ascii(self):
        """The source file services/calendly_service.py must decode as pure ASCII."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "services",
            "calendly_service.py",
        )
        with open(source_path, "rb") as f:
            content = f.read()
        try:
            content.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"Source file contains non-ASCII bytes: {exc}")


# ---------------------------------------------------------------------------
# B. _normalize_base_link
# ---------------------------------------------------------------------------


class TestNormalizeBaseLink(unittest.TestCase):
    """Validation and normalisation of raw base scheduling URLs."""

    def test_valid_urls_normalised(self):
        cases = [
            (' "https://calendly.com/x/y" ', "https://calendly.com/x/y"),
            (" 'https://calendly.com/x/y' ", "https://calendly.com/x/y"),
            ("http://calendly.com/x/y", "https://calendly.com/x/y"),
            ("calendly.com/x/y", "https://calendly.com/x/y"),
            ("calendly.com/x/y?a=1&b=2#sec", "https://calendly.com/x/y?a=1&b=2#sec"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_base_link(raw), expected)

    def test_invalid_inputs_return_none(self):
        invalid_cases = [
            "https://calendly.com",
            "https://calendly.com/",
            "https://u:p@calendly.com/x/y",
            "not a url",
            "ftp://calendly.com/x/y",
            "https://localhost/x/y",
            "https://calendly.com/x y/z",
            "",
            "   ",
            None,
            123,
            [],
            {},
        ]
        for raw in invalid_cases:
            with self.subTest(raw=raw):
                self.assertIsNone(_normalize_base_link(raw))


# ---------------------------------------------------------------------------
# C. get_base_link
# ---------------------------------------------------------------------------


class TestGetBaseLink(unittest.TestCase):
    """Reading CALENDLY_LINK from environment and graceful fallback."""

    def test_unset_and_blank_produce_none_silently(self):
        for val in [None, "", "   "]:
            with self.subTest(val=val):
                env = {} if val is None else {"CALENDLY_LINK": val}
                with patch.dict(os.environ, env, clear=True):
                    with self.assertNoLogs("services.calendly_service", level="WARNING"):
                        self.assertIsNone(get_base_link())

    def test_invalid_logs_warning_without_leaking_value(self):
        secret_bad = "invalid-url-with-secrets"
        with patch.dict(os.environ, {"CALENDLY_LINK": secret_bad}, clear=True):
            with self.assertLogs("services.calendly_service", level="WARNING") as cm:
                res = get_base_link()
            self.assertIsNone(res)
            self.assertEqual(len(cm.output), 1)
            self.assertIn(
                "CALENDLY_LINK is set but is not a valid scheduling URL",
                cm.output[0],
            )
            self.assertNotIn(secret_bad, cm.output[0])

    def test_valid_returns_normalised(self):
        with patch.dict(
            os.environ,
            {"CALENDLY_LINK": "calendly.com/team/intro"},
            clear=True,
        ):
            self.assertEqual(get_base_link(), "https://calendly.com/team/intro")


# ---------------------------------------------------------------------------
# D. build_booking_link (email channel defaults)
# ---------------------------------------------------------------------------


class TestBuildBookingLinkEmail(unittest.TestCase):
    """Email channel pre-fill formatting and UTM injection."""

    def test_email_defaults(self):
        lead = _make_lead()
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            expected = (
                BASE
                + "?name=Sarah%20Mitchell&email=sarah.mitchell%40example.com"
                "&utm_source=lead_system&utm_medium=email&utm_campaign=ai_followup"
            )
            actual = build_booking_link(lead, channel="email")
            self.assertEqual(actual, expected)


# ---------------------------------------------------------------------------
# E. build_booking_link (SMS channel)
# ---------------------------------------------------------------------------


class TestBuildBookingLinkSms(unittest.TestCase):
    """SMS channel privacy: no personal data, minimal UTM."""

    def test_sms_channel(self):
        lead = _make_lead()
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            url = build_booking_link(lead, channel="sms")
            self.assertEqual(url, BASE + "?utm_medium=sms")
            qs = parse_qs(urlsplit(url).query)
            self.assertNotIn("name", qs)
            self.assertNotIn("email", qs)

            url_no_utm = build_booking_link(lead, channel="sms", include_utm=False)
            self.assertEqual(url_no_utm, BASE)
            qs_no_utm = parse_qs(urlsplit(url_no_utm).query)
            self.assertNotIn("name", qs_no_utm)
            self.assertNotIn("email", qs_no_utm)


# ---------------------------------------------------------------------------
# F. CALENDLY_NAME_FIELDS (single vs split mode)
# ---------------------------------------------------------------------------


class TestCalendlyNameFields(unittest.TestCase):
    """First/last name splitting configuration."""

    def test_split_mode(self):
        lead = _make_lead()
        for mode in ["split", " SPLIT "]:
            with self.subTest(mode=mode):
                with patch.dict(
                    os.environ,
                    {"CALENDLY_LINK": BASE, "CALENDLY_NAME_FIELDS": mode},
                    clear=True,
                ):
                    url = build_booking_link(lead, channel="email")
                    qs = parse_qs(urlsplit(url).query)
                    self.assertEqual(qs.get("first_name"), ["Sarah"])
                    self.assertEqual(qs.get("last_name"), ["Mitchell"])
                    self.assertNotIn("name", qs)

    def test_one_word_name_in_split_mode(self):
        lead_one = _make_lead(name="Sarah")
        with patch.dict(
            os.environ,
            {"CALENDLY_LINK": BASE, "CALENDLY_NAME_FIELDS": "split"},
            clear=True,
        ):
            url = build_booking_link(lead_one, channel="email")
            qs = parse_qs(urlsplit(url).query)
            self.assertEqual(qs.get("first_name"), ["Sarah"])
            self.assertNotIn("last_name", qs)
            self.assertNotIn("name", qs)

    def test_other_values_default_to_single_mode(self):
        lead = _make_lead()
        for mode in ["single", " SINGLE ", "invalid", "", "other"]:
            with self.subTest(mode=mode):
                with patch.dict(
                    os.environ,
                    {"CALENDLY_LINK": BASE, "CALENDLY_NAME_FIELDS": mode},
                    clear=True,
                ):
                    url = build_booking_link(lead, channel="email")
                    qs = parse_qs(urlsplit(url).query)
                    self.assertEqual(qs.get("name"), ["Sarah Mitchell"])
                    self.assertNotIn("first_name", qs)
                    self.assertNotIn("last_name", qs)


# ---------------------------------------------------------------------------
# G. Special characters round-trip
# ---------------------------------------------------------------------------


class TestSpecialCharacters(unittest.TestCase):
    """Safe URL encoding of unicode names, apostrophes, ampersands, and pluses."""

    def test_special_characters_round_trip(self):
        cases = [
            ("Jos\u00e9 O'Connor", "sarah@example.com"),
            ("Smith & Jones", "sarah@example.com"),
            ("\u674e\u660e", "sarah@example.com"),
            ("Zo\u00eb \U0001F600", "sarah@example.com"),
            ("Sarah Mitchell", "a+b@example.com"),
        ]
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            for name, email in cases:
                with self.subTest(name=name, email=email):
                    lead = SimpleNamespace(name=name, email=email)
                    url = build_booking_link(lead, channel="email")
                    query = urlsplit(url).query
                    self.assertNotIn(" ", query)
                    qs = parse_qs(query)
                    self.assertEqual(qs.get("name"), [name])
                    self.assertEqual(qs.get("email"), [email])

    def test_literal_encodings(self):
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            lead_jose = SimpleNamespace(
                name="Jos\u00e9 O'Connor",
                email="sarah.mitchell@example.com",
            )
            url_jose = build_booking_link(lead_jose, channel="email")
            self.assertIn("Jos%C3%A9%20O%27Connor", url_jose)

            lead_plus = SimpleNamespace(
                name="Sarah Mitchell",
                email="a+b@example.com",
            )
            url_plus = build_booking_link(lead_plus, channel="email")
            self.assertIn("a%2Bb%40example.com", url_plus)


# ---------------------------------------------------------------------------
# H. Odd data and edge cases
# ---------------------------------------------------------------------------


class TestOddData(unittest.TestCase):
    """Handling missing, blank, oversized, or malformed lead data."""

    def test_blank_and_missing_lead_attributes(self):
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            # Blank name -> no name param
            l1 = SimpleNamespace(name="   ", email="sarah.mitchell@example.com")
            qs1 = parse_qs(urlsplit(build_booking_link(l1, channel="email")).query)
            self.assertNotIn("name", qs1)
            self.assertIn("email", qs1)

            # Blank email -> no email param
            l2 = SimpleNamespace(name="Sarah Mitchell", email="   ")
            qs2 = parse_qs(urlsplit(build_booking_link(l2, channel="email")).query)
            self.assertIn("name", qs2)
            self.assertNotIn("email", qs2)

            # Both blank -> UTM only
            l3 = SimpleNamespace(name="", email="")
            qs3 = parse_qs(urlsplit(build_booking_link(l3, channel="email")).query)
            self.assertNotIn("name", qs3)
            self.assertNotIn("email", qs3)
            self.assertIn("utm_source", qs3)
            self.assertIn("utm_medium", qs3)
            self.assertIn("utm_campaign", qs3)

            # Control characters / newlines / tabs collapse to spaces
            l4 = SimpleNamespace(
                name="Sarah\n\t\x00Mitchell",
                email="sarah.mitchell@example.com",
            )
            qs4 = parse_qs(urlsplit(build_booking_link(l4, channel="email")).query)
            self.assertEqual(qs4.get("name"), ["Sarah Mitchell"])

            # 300-character name capped at 100 characters before encoding
            l5 = SimpleNamespace(name="a" * 300, email="sarah.mitchell@example.com")
            qs5 = parse_qs(urlsplit(build_booking_link(l5, channel="email")).query)
            self.assertEqual(qs5.get("name"), ["a" * 100])

            # Unknown channel raises ValueError
            with self.assertRaises(ValueError):
                build_booking_link(l1, channel="carrier_pigeon")

            # Incomplete object missing attributes entirely does not raise
            incomplete = object()
            url_inc = build_booking_link(incomplete, channel="email")
            self.assertIsNotNone(url_inc)
            qs_inc = parse_qs(urlsplit(url_inc).query)
            self.assertNotIn("name", qs_inc)
            self.assertNotIn("email", qs_inc)


# ---------------------------------------------------------------------------
# I. CALENDLY_UTM_CAMPAIGN sanitisation
# ---------------------------------------------------------------------------


class TestUtmCampaign(unittest.TestCase):
    """Sanitising and capping CALENDLY_UTM_CAMPAIGN."""

    def test_utm_campaign_sanitisation(self):
        lead = _make_lead()
        cases = [
            (None, "ai_followup"),
            ("Spring Sale!", "Spring_Sale"),
            ("   ", "ai_followup"),
            ("a" * 80, "a" * 50),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw, expected=expected):
                env = {"CALENDLY_LINK": BASE}
                if raw is not None:
                    env["CALENDLY_UTM_CAMPAIGN"] = raw
                with patch.dict(os.environ, env, clear=True):
                    url = build_booking_link(lead, channel="email")
                    qs = parse_qs(urlsplit(url).query)
                    self.assertEqual(qs.get("utm_campaign"), [expected])


# ---------------------------------------------------------------------------
# J. Base link with existing query parameters
# ---------------------------------------------------------------------------


class TestBaseWithQuery(unittest.TestCase):
    """Preserving operator query parameters while overriding pre-fill keys."""

    def test_operator_params_preserved(self):
        lead = _make_lead()
        base_with_params = BASE + "?hide_gdpr_banner=1&utm_campaign=spring"
        with patch.dict(
            os.environ,
            {"CALENDLY_LINK": base_with_params, "CALENDLY_UTM_CAMPAIGN": "ai_followup"},
            clear=True,
        ):
            url = build_booking_link(lead, channel="email")
            qs = parse_qs(urlsplit(url).query)
            self.assertEqual(qs.get("hide_gdpr_banner"), ["1"])
            self.assertEqual(qs.get("utm_campaign"), ["spring"])
            self.assertEqual(qs.get("utm_source"), ["lead_system"])
            self.assertEqual(qs.get("utm_medium"), ["email"])

    def test_prefill_overwrites_existing_keys(self):
        lead = _make_lead()
        base_with_old_name = BASE + "?name=Old"
        with patch.dict(os.environ, {"CALENDLY_LINK": base_with_old_name}, clear=True):
            url = build_booking_link(lead, channel="email")
            qs = parse_qs(urlsplit(url).query)
            self.assertEqual(qs.get("name"), ["Sarah Mitchell"])


# ---------------------------------------------------------------------------
# K. strip_urls
# ---------------------------------------------------------------------------


class TestStripUrls(unittest.TestCase):
    """Removal of hallucinated URLs and Calendly domain references."""

    def test_strip_urls(self):
        text = (
            "Here is https://x.com/a?b=1 and http://insecure.org/test\n"
            "Also check www.x.com/y and calendly.com/agent for details."
        )
        cleaned = strip_urls(text)
        self.assertNotIn("https://x.com/a?b=1", cleaned)
        self.assertNotIn("http://insecure.org/test", cleaned)
        self.assertNotIn("www.x.com/y", cleaned)
        self.assertNotIn("calendly.com/agent", cleaned)
        self.assertIn("Here is and", cleaned)
        self.assertIn("\nAlso check and for details.", cleaned)

    def test_non_string_returns_empty(self):
        self.assertEqual(strip_urls(None), "")
        self.assertEqual(strip_urls(123), "")


# ---------------------------------------------------------------------------
# L. inject_link_into_email_body
# ---------------------------------------------------------------------------


class TestInjectLinkIntoEmailBody(unittest.TestCase):
    """Deterministic token replacement and fallback handling for email."""

    def test_inject_link_into_email_body(self):
        link = "https://calendly.com/team/view?utm_source=lead_system&utm_medium=email"

        # Token replaced, link sits alone on line, token gone
        body_with_token = f"Hi Sarah,\n\nBook viewing:\n{LINK_TOKEN}\n\nThanks"
        res1 = inject_link_into_email_body(body_with_token, link)
        self.assertNotIn(LINK_TOKEN, res1)
        self.assertIn(f"\n{link}\n", res1)

        # Two tokens -> link once, all tokens removed
        body_two_tokens = f"First {LINK_TOKEN} then second {LINK_TOKEN} end."
        res2 = inject_link_into_email_body(body_two_tokens, link)
        self.assertNotIn(LINK_TOKEN, res2)
        self.assertEqual(res2.count(link), 1)

        # Model-written URL is removed while the real link stays
        body_with_hallucination = f"Check https://evil.com/fake\n{LINK_TOKEN}"
        res3 = inject_link_into_email_body(body_with_hallucination, link)
        self.assertNotIn("https://evil.com/fake", res3)
        self.assertIn(link, res3)

        # Token missing -> CTA fallback + link appended
        body_no_token = "Hi Sarah,\n\nLooking forward to speaking."
        res4 = inject_link_into_email_body(body_no_token, link)
        self.assertIn(_EMAIL_CTA_FALLBACK, res4)
        self.assertIn(link, res4)

        # Link None -> tokens removed and no URL left
        body_with_urls = f"Contact us {LINK_TOKEN} or visit https://bad.com"
        res5 = inject_link_into_email_body(body_with_urls, None)
        self.assertNotIn(LINK_TOKEN, res5)
        self.assertNotIn("https://bad.com", res5)

        # Runs of 3+ newlines collapsed to 2
        body_many_newlines = f"Line 1\n\n\n\nLine 2\n\n\n{LINK_TOKEN}\n\n\n\nLine 3"
        res6 = inject_link_into_email_body(body_many_newlines, link)
        self.assertNotIn("\n\n\n", res6)

        # Ampersand in the link untouched
        self.assertIn("&", link)
        self.assertIn(link, res1)


# ---------------------------------------------------------------------------
# M. plan_sms
# ---------------------------------------------------------------------------


class TestPlanSms(unittest.TestCase):
    """SMS suffix and budget planning according to link length."""

    def test_plan_sms_base(self):
        lead = _make_lead()
        with patch.dict(os.environ, {"CALENDLY_LINK": BASE}, clear=True):
            plan = plan_sms(lead)
            expected_link = BASE + "?utm_medium=sms"
            expected_suffix = " " + expected_link + " " + SMS_OPT_OUT
            self.assertEqual(plan.link, expected_link)
            self.assertEqual(plan.suffix, expected_suffix)
            self.assertEqual(plan.text_budget, 68)

    def test_plan_sms_base_85_chars(self):
        lead = _make_lead()
        prefix = "https://calendly.com/"
        b85 = prefix + "a" * (85 - len(prefix))
        self.assertEqual(len(b85), 85)
        with patch.dict(os.environ, {"CALENDLY_LINK": b85}, clear=True):
            plan = plan_sms(lead)
            self.assertEqual(plan.link, b85)
            self.assertNotIn("utm_medium", plan.link)
            self.assertEqual(plan.suffix, " " + b85 + " " + SMS_OPT_OUT)
            self.assertEqual(plan.text_budget, 52)

    def test_plan_sms_base_100_chars(self):
        lead = _make_lead()
        prefix = "https://calendly.com/"
        b100 = prefix + "a" * (100 - len(prefix))
        self.assertEqual(len(b100), 100)
        with patch.dict(os.environ, {"CALENDLY_LINK": b100}, clear=True):
            plan = plan_sms(lead)
            self.assertIsNone(plan.link)
            self.assertEqual(plan.suffix, " " + SMS_OPT_OUT)
            self.assertEqual(plan.text_budget, 138)

    def test_plan_sms_env_unset(self):
        lead = _make_lead()
        with patch.dict(os.environ, {}, clear=True):
            plan = plan_sms(lead)
            self.assertIsNone(plan.link)
            self.assertEqual(plan.suffix, " " + SMS_OPT_OUT)
            self.assertEqual(plan.text_budget, 138)


# ---------------------------------------------------------------------------
# N. finish_sms
# ---------------------------------------------------------------------------


class TestFinishSms(unittest.TestCase):
    """Enforcement of SMS segment length, opt-out guarantee, and fallbacks."""

    def test_finish_sms_grid(self):
        lead = _make_lead()
        plan_with_link = SmsPlan(
            link=BASE + "?utm_medium=sms",
            suffix=" " + BASE + "?utm_medium=sms " + SMS_OPT_OUT,
            text_budget=68,
        )
        plan_without_link = SmsPlan(
            link=None,
            suffix=" " + SMS_OPT_OUT,
            text_budget=138,
        )

        for plan in [plan_with_link, plan_without_link]:
            for length in [0, 10, 45, 60, 100, 160, 200, 300]:
                with self.subTest(has_link=bool(plan.link), length=length):
                    raw_text = ("word " * ((length // 5) + 2))[:length]
                    msg = finish_sms(raw_text, plan, lead)
                    self.assertLessEqual(len(msg), SMS_MAX_CHARS)
                    self.assertTrue(msg.endswith(SMS_OPT_OUT))
                    self.assertEqual(msg.count(SMS_OPT_OUT), 1)
                    self.assertNotIn(LINK_TOKEN, msg)
                    if plan.link:
                        self.assertEqual(msg.count(plan.link), 1)
                        expected_suffix = " " + plan.link + " " + SMS_OPT_OUT
                        self.assertTrue(msg.endswith(expected_suffix))

    def test_finish_sms_cleanups_and_fallbacks(self):
        lead = _make_lead()
        plan = SmsPlan(
            link=BASE + "?utm_medium=sms",
            suffix=" " + BASE + "?utm_medium=sms " + SMS_OPT_OUT,
            text_budget=68,
        )

        # Model-added URL and opt-out removed
        input_text = "Check https://evil.com/x Reply STOP to opt out. Book viewing:"
        msg_cleaned = finish_sms(input_text, plan, lead)
        self.assertNotIn("https://evil.com", msg_cleaned)
        self.assertEqual(msg_cleaned.count(SMS_OPT_OUT), 1)

        # Unicode ellipsis \u2026 becomes ...
        msg_ellipsis = finish_sms("Wait\u2026 deals coming", plan, lead)
        self.assertNotIn("\u2026", msg_ellipsis)
        self.assertIn("...", msg_ellipsis)

        # Empty text -> fallback with lead first name
        msg_empty = finish_sms("    ", plan, lead)
        self.assertIn("Hi Sarah, an agent will contact you shortly.", msg_empty)

        # Over-long text is cut at a word boundary
        overlong = (
            "Quick brown fox jumps over lazy dog repeatedly until "
            "the buffer is completely full"
        )
        msg_cut = finish_sms(overlong, plan, lead)
        self.assertLessEqual(len(msg_cut), SMS_MAX_CHARS)
        body_part = msg_cut[: -len(plan.suffix)]
        self.assertFalse(body_part.endswith(" "))
        self.assertFalse(body_part.endswith(","))


if __name__ == "__main__":
    unittest.main()
