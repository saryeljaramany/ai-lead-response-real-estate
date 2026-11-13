"""
tests/test_phone_service.py

Unit tests for services/phone_service.py.

Rules enforced:
- Only unittest and unittest.mock.
- No network access, no .env file read.
- main.py is never imported.
- All environment-dependent tests use patch.dict(os.environ, ..., clear=True).
- Table-driven tests use subTest.
- Pure ASCII and saved as UTF-8 without BOM.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import services.phone_service as ps


class TestPhoneService(unittest.TestCase):
    """Automated test suite for services/phone_service.py."""

    # -----------------------------------------------------------------------
    # A. Import safety & ASCII source
    # -----------------------------------------------------------------------

    def test_import_safety(self) -> None:
        """Reloading services.phone_service with an empty environment must not raise."""
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(ps)

    def test_source_is_ascii(self) -> None:
        """Source code of services/phone_service.py must decode as pure ASCII."""
        source_path = (
            Path(__file__).resolve().parent.parent / "services" / "phone_service.py"
        )
        content = source_path.read_bytes()
        try:
            content.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"services/phone_service.py is not pure ASCII: {exc}")

    # -----------------------------------------------------------------------
    # B. _default_region tests
    # -----------------------------------------------------------------------

    def test_default_region_unset_and_blank_no_log(self) -> None:
        """Unset or blank DEFAULT_COUNTRY_CODE returns None silently with no warning log."""
        test_cases = [
            ("unset", None),
            ("empty", ""),
            ("spaces", "   "),
            ("tabs_newlines", "\t\n  "),
        ]
        logger = logging.getLogger("services.phone_service")
        for label, val in test_cases:
            with self.subTest(case=label):
                env = {} if val is None else {"DEFAULT_COUNTRY_CODE": val}
                with patch.dict(os.environ, env, clear=True):
                    with patch.object(logger, "warning") as mock_warn:
                        result = ps._default_region()
                        self.assertIsNone(result)
                        mock_warn.assert_not_called()

    def test_default_region_valid_iso_codes(self) -> None:
        """Two-letter ISO codes (case-insensitive, whitespace/quotes stripped) return uppercase region."""
        valid_iso_cases = [
            ("US", "US", "US"),
            ("us", "us", "US"),
            (" it ", " it ", "IT"),
            ("quoted_GB", "'GB'", "GB"),
            ("double_quoted_it", '"it"', "IT"),
        ]
        for label, env_val, expected_region in valid_iso_cases:
            with self.subTest(case=label):
                with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": env_val}, clear=True):
                    self.assertEqual(ps._default_region(), expected_region)

    def test_default_region_valid_calling_codes(self) -> None:
        """Calling codes with optional '+' or '00' resolve to the primary region code."""
        calling_code_cases = [
            ("1", "1", "US"),
            ("+1", "+1", "US"),
            ("39", "39", "IT"),
            ("+39", "+39", "IT"),
            ("0039", "0039", "IT"),
            ("001", "001", "US"),
        ]
        for label, env_val, expected_region in calling_code_cases:
            with self.subTest(case=label):
                with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": env_val}, clear=True):
                    self.assertEqual(ps._default_region(), expected_region)

    def test_default_region_invalid_values_log_warning(self) -> None:
        """Invalid values return None and emit exactly ONE warning not leaking the value."""
        invalid_cases = [
            ("ZZ", "ZZ"),
            ("99999", "99999"),
            ("abc", "abc"),
            ("USA", "USA"),
        ]
        expected_warning = (
            "DEFAULT_COUNTRY_CODE is set but is not a valid ISO country code or calling code; "
            "national phone numbers will be rejected."
        )

        for label, env_val in invalid_cases:
            with self.subTest(case=label):
                with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": env_val}, clear=True):
                    with self.assertLogs("services.phone_service", level="WARNING") as cm:
                        result = ps._default_region()
                        self.assertIsNone(result)
                        self.assertEqual(len(cm.output), 1)
                        log_msg = cm.output[0]
                        self.assertIn(expected_warning, log_msg)
                        self.assertNotIn(env_val, log_msg)

    # -----------------------------------------------------------------------
    # C. International numbers
    # -----------------------------------------------------------------------

    def test_international_valid_numbers(self) -> None:
        """Valid international numbers (prefixed with '+' or '00') normalise to E.164."""
        valid_cases = [
            ("+1-415-555-0192", "+1-415-555-0192", "+14155550192"),
            (" +39 347 123 4567 ", " +39 347 123 4567 ", "+393471234567"),
            ("0039 347 1234567", "0039 347 1234567", "+393471234567"),
            ("+1 (415) 555-0192", "+1 (415) 555-0192", "+14155550192"),
            ("+15005550006", "+15005550006", "+15005550006"),  # Twilio magic number
        ]
        for label, raw_input, expected_e164 in valid_cases:
            with self.subTest(case=label):
                res = ps.normalize_phone(raw_input)
                self.assertEqual(res.e164, expected_e164)
                self.assertEqual(res.reason, "international")

    def test_international_invalid_numbers(self) -> None:
        """Malformed international numbers fail E.164 pattern and return (None, 'invalid')."""
        invalid_cases = [
            ("+0123456789", "+0123456789"),        # Leading zero after +
            ("+123", "+123"),                      # Too short (< 8 total digits)
            ("+1415555019299999", "+1415555019299999"),  # Too long (> 15 total digits)
        ]
        for label, raw_input in invalid_cases:
            with self.subTest(case=label):
                res = ps.normalize_phone(raw_input)
                self.assertIsNone(res.e164)
                self.assertEqual(res.reason, "invalid")

    # -----------------------------------------------------------------------
    # D. National numbers with region
    # -----------------------------------------------------------------------

    def test_national_numbers_with_us_region(self) -> None:
        """National numbers with DEFAULT_COUNTRY_CODE=US or 1 format correctly to E.164."""
        us_inputs = [
            ("(415) 555-0192", "(415) 555-0192"),
            ("415-555-0192", "415-555-0192"),
            ("1 415 555 0192", "1 415 555 0192"),
        ]
        for country_setting in ["US", "1"]:
            with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": country_setting}, clear=True):
                for label, raw_input in us_inputs:
                    with self.subTest(setting=country_setting, case=label):
                        res = ps.normalize_phone(raw_input)
                        self.assertEqual(res.e164, "+14155550192")
                        self.assertEqual(res.reason, "national")

    def test_national_numbers_with_it_region(self) -> None:
        """National numbers with DEFAULT_COUNTRY_CODE=IT or 39 format correctly to E.164."""
        for country_setting in ["IT", "39"]:
            with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": country_setting}, clear=True):
                with self.subTest(setting=country_setting):
                    res = ps.normalize_phone("347 1234567")
                    self.assertEqual(res.e164, "+393471234567")
                    self.assertEqual(res.reason, "national")

    # -----------------------------------------------------------------------
    # E. National without region
    # -----------------------------------------------------------------------

    def test_national_without_region_returns_needs_country_code(self) -> None:
        """National digits without DEFAULT_COUNTRY_CODE return (None, 'needs_country_code')."""
        inputs = [
            "4155550192",
            "(415) 555-0192",
            "347 1234567",
        ]
        with patch.dict(os.environ, {}, clear=True):
            for raw_input in inputs:
                with self.subTest(input=raw_input):
                    res = ps.normalize_phone(raw_input)
                    self.assertIsNone(res.e164)
                    self.assertEqual(res.reason, "needs_country_code")

    # -----------------------------------------------------------------------
    # F. Invalid inputs
    # -----------------------------------------------------------------------

    def test_invalid_national_numbers_with_us_region(self) -> None:
        """Invalid national numbers under US region return None for e164 with invalid reason."""
        with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": "US"}, clear=True):
            for raw_input in ["123", "555-0192"]:
                with self.subTest(input=raw_input):
                    res = ps.normalize_phone(raw_input)
                    self.assertIsNone(res.e164)
                    self.assertIn(res.reason, ["invalid", "invalid_for_region"])

    def test_non_digit_and_malformed_inputs(self) -> None:
        """Non-digit strings, empty strings, non-strings return (None, 'invalid')."""
        cases = [
            ("abc", "abc"),
            ("empty", ""),
            ("none", None),
            ("int", 123),
            ("list", []),
        ]
        with patch.dict(os.environ, {"DEFAULT_COUNTRY_CODE": "US"}, clear=True):
            for label, raw_input in cases:
                with self.subTest(case=label):
                    res = ps.normalize_phone(raw_input)
                    self.assertIsNone(res.e164)
                    self.assertEqual(res.reason, "invalid")

    # -----------------------------------------------------------------------
    # G. Robustness against unusual inputs
    # -----------------------------------------------------------------------

    def test_weird_inputs_never_raise(self) -> None:
        """Emoji, ultra-long strings, and separator-only inputs never raise an exception."""
        weird_cases = [
            ("emoji", "\U0001F600"),
            ("5000_chars_alpha", "a" * 5000),
            ("5000_chars_digits", "9" * 5000),
            ("only_separators", "   ---...((()))   "),
        ]
        for label, raw_input in weird_cases:
            with self.subTest(case=label):
                res = ps.normalize_phone(raw_input)
                self.assertIsInstance(res, ps.PhoneNormalization)

    # -----------------------------------------------------------------------
    # H. Call-time environment reading
    # -----------------------------------------------------------------------

    def test_environment_read_at_call_time(self) -> None:
        """Changing DEFAULT_COUNTRY_CODE between calls in a single test reflects immediately."""
        with patch.dict(os.environ, {}, clear=True):
            res_no_env = ps.normalize_phone("(415) 555-0192")
            self.assertIsNone(res_no_env.e164)
            self.assertEqual(res_no_env.reason, "needs_country_code")

            os.environ["DEFAULT_COUNTRY_CODE"] = "US"
            res_us = ps.normalize_phone("(415) 555-0192")
            self.assertEqual(res_us.e164, "+14155550192")
            self.assertEqual(res_us.reason, "national")

            os.environ["DEFAULT_COUNTRY_CODE"] = "IT"
            res_it = ps.normalize_phone("(415) 555-0192")
            self.assertIsNone(res_it.e164)
            self.assertEqual(res_it.reason, "invalid_for_region")


if __name__ == "__main__":
    unittest.main()
