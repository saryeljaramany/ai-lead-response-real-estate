"""
tests/test_twilio_service.py

Unit tests for services/twilio_service.py.

Rules enforced:
- No network access.
- No .env file read; env vars are injected via patch.dict(os.environ, ..., clear=True).
- main.py is never imported.
- TwilioRestException constructor confirmed via inspect.signature before use.
"""

import importlib
import inspect
import math
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, patch as mock_patch

# ---------------------------------------------------------------------------
# Confirm TwilioRestException constructor before the test suite runs
# ---------------------------------------------------------------------------
from twilio.base.exceptions import TwilioRestException as _TRE

_TRE_SIG = inspect.signature(_TRE.__init__)
# Expected: (self, status, uri, msg='', code=None, method='GET', details=None)
_TRE_PARAMS = list(_TRE_SIG.parameters.keys())
assert "status" in _TRE_PARAMS, "TwilioRestException.__init__ missing 'status' param"
assert "uri" in _TRE_PARAMS, "TwilioRestException.__init__ missing 'uri' param"
assert "code" in _TRE_PARAMS, "TwilioRestException.__init__ missing 'code' param"


def _make_twilio_exc(status: int, code: int | None = None, msg: str = "test error"):
    """Construct a TwilioRestException using its confirmed signature."""
    return _TRE(status=status, uri="https://api.twilio.com/test", msg=msg, code=code)


# ---------------------------------------------------------------------------
# Import the module under a clean environment so import-time safety is proven
# ---------------------------------------------------------------------------
def _import_service_clean():
    """Import (or reload) twilio_service with an empty environment."""
    with patch.dict(os.environ, {}, clear=True):
        import services.twilio_service as svc
        importlib.reload(svc)
    return svc


# ---------------------------------------------------------------------------
# A. Import safety
# ---------------------------------------------------------------------------
class TestImportSafety(unittest.TestCase):

    def test_import_with_empty_env_does_not_raise(self):
        """Importing with an empty environment must never raise."""
        try:
            with patch.dict(os.environ, {}, clear=True):
                import services.twilio_service as svc
                importlib.reload(svc)
        except Exception as exc:
            self.fail(f"Import raised {type(exc).__name__}: {exc}")

    def test_source_file_is_ascii(self):
        """The source file must be decodable as pure ASCII."""
        import services.twilio_service as svc
        src_path = inspect.getfile(svc)
        with open(src_path, "rb") as f:
            raw = f.read()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:
            self.fail(f"twilio_service.py is not pure ASCII: {exc}")


# ---------------------------------------------------------------------------
# Helpers shared by the remaining tests
# ---------------------------------------------------------------------------
def _svc():
    """Return the module, reloading it under a clean environment."""
    return _import_service_clean()


# ---------------------------------------------------------------------------
# B. _normalize_e164
# ---------------------------------------------------------------------------
class TestNormalizeE164(unittest.TestCase):

    def test_valid_numbers(self):
        cases = [
            ("+1-415-555-0192", "+14155550192"),
            (" +39 347 123 4567 ", "+393471234567"),
            ("0039 347 1234567", "+393471234567"),
            ("+1 (415) 555-0192", "+14155550192"),
        ]
        svc = _svc()
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(svc._normalize_e164(raw), expected)

    def test_invalid_numbers_raise_value_error(self):
        invalids = [
            "415-555-0192",          # no country code
            "12345678",              # no leading +
            "+0123456789",           # leading zero after +
            "+123",                  # too short (< 8 digits after +)
            "abc",                   # non-numeric
            "",                      # empty
            "+1415555019299999",     # more than 15 digits
        ]
        svc = _svc()
        for raw in invalids:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    svc._normalize_e164(raw)


# ---------------------------------------------------------------------------
# C. _sanitize_body
# ---------------------------------------------------------------------------
class TestSanitizeBody(unittest.TestCase):

    def test_curly_quotes_replaced(self):
        svc = _svc()
        result = svc._sanitize_body("\u2018hello\u2019 and \u201chello\u201d")
        self.assertNotIn("\u2018", result)
        self.assertNotIn("\u2019", result)
        self.assertNotIn("\u201c", result)
        self.assertNotIn("\u201d", result)
        self.assertIn("'", result)
        self.assertIn('"', result)

    def test_en_em_dashes_replaced(self):
        svc = _svc()
        result = svc._sanitize_body("a\u2013b\u2014c")
        self.assertNotIn("\u2013", result)
        self.assertNotIn("\u2014", result)
        self.assertIn("-", result)

    def test_ellipsis_replaced(self):
        svc = _svc()
        result = svc._sanitize_body("wait\u2026")
        self.assertNotIn("\u2026", result)
        self.assertIn("...", result)

    def test_nbsp_replaced_with_space(self):
        svc = _svc()
        result = svc._sanitize_body("hello\u00a0world")
        self.assertNotIn("\u00a0", result)
        self.assertIn(" ", result)

    def test_zero_width_characters_deleted(self):
        svc = _svc()
        result = svc._sanitize_body("hello\u200bworld")
        self.assertNotIn("\u200b", result)

    def test_only_zero_width_raises_value_error(self):
        svc = _svc()
        with self.assertRaises(ValueError):
            svc._sanitize_body("\u200b\u200b\ufeff")


# ---------------------------------------------------------------------------
# D. _analyze_body
# ---------------------------------------------------------------------------
class TestAnalyzeBody(unittest.TestCase):

    def test_gsm7_160_chars_one_segment(self):
        svc = _svc()
        enc, units, segs = svc._analyze_body("a" * 160)
        self.assertEqual(enc, "GSM-7")
        self.assertEqual(units, 160)
        self.assertEqual(segs, 1)

    def test_gsm7_161_chars_two_segments(self):
        svc = _svc()
        _, _, segs = svc._analyze_body("a" * 161)
        self.assertEqual(segs, 2)

    def test_gsm7_306_chars_two_segments(self):
        svc = _svc()
        _, _, segs = svc._analyze_body("a" * 306)
        self.assertEqual(segs, 2)

    def test_gsm7_307_chars_three_segments(self):
        svc = _svc()
        _, _, segs = svc._analyze_body("a" * 307)
        self.assertEqual(segs, 3)

    def test_euro_sign_80_one_segment(self):
        # € is in the GSM-7 extension table, so counts as 2 units each → 160 total
        svc = _svc()
        enc, units, segs = svc._analyze_body("\u20ac" * 80)
        self.assertEqual(enc, "GSM-7")
        self.assertEqual(units, 160)
        self.assertEqual(segs, 1)

    def test_euro_sign_81_two_segments(self):
        svc = _svc()
        enc, units, segs = svc._analyze_body("\u20ac" * 81)
        self.assertEqual(enc, "GSM-7")
        self.assertEqual(units, 162)
        self.assertEqual(segs, 2)

    def test_cafe_is_gsm7(self):
        svc = _svc()
        enc, _, _ = svc._analyze_body("caf\u00e9")
        self.assertEqual(enc, "GSM-7")

    def test_ucs2_70_chars_one_segment(self):
        # ê (U+00EA) is not in GSM-7
        svc = _svc()
        enc, units, segs = svc._analyze_body("\u00ea" * 70)
        self.assertEqual(enc, "UCS-2")
        self.assertEqual(segs, 1)

    def test_ucs2_71_chars_two_segments(self):
        svc = _svc()
        enc, _, segs = svc._analyze_body("\u00ea" * 71)
        self.assertEqual(enc, "UCS-2")
        self.assertEqual(segs, 2)

    def test_emoji_35_one_segment(self):
        # 😀 (U+1F600) is a supplementary character: 2 UTF-16 code units each → 70 total
        svc = _svc()
        enc, units, segs = svc._analyze_body("\U0001F600" * 35)
        self.assertEqual(enc, "UCS-2")
        self.assertEqual(units, 70)
        self.assertEqual(segs, 1)

    def test_emoji_36_two_segments(self):
        svc = _svc()
        enc, _, segs = svc._analyze_body("\U0001F600" * 36)
        self.assertEqual(enc, "UCS-2")
        self.assertEqual(segs, 2)

    def test_unsanitised_curly_apostrophe_is_ucs2(self):
        # it\u2019s — right single quotation mark is NOT in GSM-7
        svc = _svc()
        enc, _, _ = svc._analyze_body("it\u2019s")
        self.assertEqual(enc, "UCS-2")


# ---------------------------------------------------------------------------
# E. _is_dry_run
# ---------------------------------------------------------------------------
class TestIsDryRun(unittest.TestCase):

    def _check(self, env_value, expected: bool, label: str):
        svc = _svc()
        env = {} if env_value is None else {"TWILIO_DRY_RUN": env_value}
        with patch.dict(os.environ, env, clear=True):
            result = svc._is_dry_run()
        self.assertEqual(result, expected, msg=f"TWILIO_DRY_RUN={label!r} -> expected {expected}")

    def test_unset_returns_true(self):
        self._check(None, True, "unset")

    def test_empty_string_returns_true(self):
        self._check("", True, "")

    def test_true_returns_true(self):
        self._check("true", True, "true")

    def test_yes_returns_true(self):
        self._check("yes", True, "yes")

    def test_garbage_returns_true(self):
        self._check("garbage", True, "garbage")

    def test_false_returns_false(self):
        self._check("false", False, "false")

    def test_FALSE_with_spaces_returns_false(self):
        self._check(" FALSE ", False, " FALSE ")

    def test_zero_returns_false(self):
        self._check("0", False, "0")

    def test_no_returns_false(self):
        self._check("no", False, "no")

    def test_off_returns_false(self):
        self._check("off", False, "off")


# ---------------------------------------------------------------------------
# F. send_sms — dry-run path
# ---------------------------------------------------------------------------
class TestSendSmsDryRun(unittest.TestCase):

    def test_dry_run_does_not_call_get_client(self):
        """
        In dry-run mode _get_client must never be called.
        Patch it to raise AssertionError to prove it.
        """
        import services.twilio_service as svc
        importlib.reload(svc)

        from models import SMSPayload
        payload = SMSPayload(
            to_phone="+1-415-555-0192",
            message="Hi Sarah, an agent will contact you shortly. Reply STOP to opt out.",
            lead_score=87,
        )

        def _boom():
            raise AssertionError("_get_client must not be called in dry-run")

        env = {}  # TWILIO_DRY_RUN unset → dry-run is active
        with patch.dict(os.environ, env, clear=True):
            with mock_patch.object(svc, "_get_client", side_effect=_boom):
                result = svc.send_sms(payload)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["status"], "dry_run")
        self.assertIsNone(result["message_id"])
        self.assertEqual(result["to"], "+14155550192")

    def test_dry_run_body_with_smart_chars_is_gsm7(self):
        """
        A body with curly apostrophe + em-dash is sanitised → GSM-7.
        """
        import services.twilio_service as svc
        importlib.reload(svc)

        from models import SMSPayload
        # \u2019 = right single quote, \u2014 = em dash; both map to ASCII
        payload = SMSPayload(
            to_phone="+14155550192",
            message="It\u2019s ready\u2014call now please ok.",
            lead_score=50,
        )

        env = {}
        with patch.dict(os.environ, env, clear=True):
            with mock_patch.object(svc, "_get_client", side_effect=AssertionError):
                result = svc.send_sms(payload)

        self.assertEqual(result["encoding"], "GSM-7")


# ---------------------------------------------------------------------------
# G. send_sms — live path
# ---------------------------------------------------------------------------
class TestSendSmsLive(unittest.TestCase):

    def _live_env(self, **extra):
        base = {
            "TWILIO_DRY_RUN": "false",
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tokentest",
            "TWILIO_PHONE_NUMBER": "+14155550199",
        }
        base.update(extra)
        return base

    def _make_mock_message(self):
        msg = MagicMock()
        msg.sid = "SM123"
        msg.status = "queued"
        return msg

    def test_live_with_phone_number_uses_from_(self):
        import services.twilio_service as svc
        importlib.reload(svc)

        from models import SMSPayload
        payload = SMSPayload(to_phone="+14155550192", message="Hello live", lead_score=70)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_message()

        with patch.dict(os.environ, self._live_env(), clear=True):
            with mock_patch.object(svc, "_get_client", return_value=mock_client):
                result = svc.send_sms(payload)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["to"], "+14155550192")
        self.assertIn("from_", call_kwargs)
        self.assertNotIn("messaging_service_sid", call_kwargs)
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["message_id"], "SM123")
        self.assertEqual(result["status"], "queued")

    def test_live_with_messaging_service_sid(self):
        import services.twilio_service as svc
        importlib.reload(svc)

        from models import SMSPayload
        payload = SMSPayload(to_phone="+14155550192", message="Hello service", lead_score=70)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_message()

        env = self._live_env(TWILIO_MESSAGING_SERVICE_SID="MG_test_sid")
        with patch.dict(os.environ, env, clear=True):
            with mock_patch.object(svc, "_get_client", return_value=mock_client):
                result = svc.send_sms(payload)

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertIn("messaging_service_sid", call_kwargs)
        self.assertEqual(call_kwargs["messaging_service_sid"], "MG_test_sid")
        self.assertNotIn("from_", call_kwargs)
        self.assertFalse(result["dry_run"])
        self.assertEqual(result["message_id"], "SM123")
        self.assertEqual(result["status"], "queued")


# ---------------------------------------------------------------------------
# H. _classify_error
# ---------------------------------------------------------------------------
class TestClassifyError(unittest.TestCase):

    def _classify(self, exc):
        import services.twilio_service as svc
        importlib.reload(svc)
        return svc._classify_error(exc)

    def test_recipient_codes(self):
        recipient_codes = [21211, 21612, 21408, 21610, 21614]
        for code in recipient_codes:
            with self.subTest(code=code):
                exc = _make_twilio_exc(status=400, code=code)
                result = self._classify(exc)
                self.assertEqual(result.category, "recipient")
                self.assertEqual(result.twilio_code, code)
                self.assertIn(str(code), str(result))

    def test_transient_codes(self):
        transient_codes = [20429, 21611]
        for code in transient_codes:
            with self.subTest(code=code):
                exc = _make_twilio_exc(status=400, code=code)
                result = self._classify(exc)
                self.assertEqual(result.category, "transient")
                self.assertEqual(result.twilio_code, code)
                self.assertIn(str(code), str(result))

    def test_transient_by_http_503_status(self):
        exc = _make_twilio_exc(status=503, code=None)
        result = self._classify(exc)
        self.assertEqual(result.category, "transient")

    def test_config_codes(self):
        config_codes = [20003, 21212, 21606, 99999]
        for code in config_codes:
            with self.subTest(code=code):
                exc = _make_twilio_exc(status=400, code=code)
                result = self._classify(exc)
                self.assertEqual(result.category, "config")
                self.assertEqual(result.twilio_code, code)
                self.assertIn(str(code), str(result))

    def test_requests_timeout_is_transient(self):
        import requests
        exc = requests.exceptions.Timeout("timed out")
        result = self._classify(exc)
        self.assertEqual(result.category, "transient")

    def test_requests_connection_error_is_transient(self):
        import requests
        exc = requests.exceptions.ConnectionError("conn refused")
        result = self._classify(exc)
        self.assertEqual(result.category, "transient")

    def test_value_error_is_config(self):
        exc = ValueError("x")
        result = self._classify(exc)
        self.assertEqual(result.category, "config")


# ---------------------------------------------------------------------------
# I. _get_client wiring
# ---------------------------------------------------------------------------
class TestGetClientWiring(unittest.TestCase):

    def test_http_client_timeout_and_client_receives_it(self):
        import services.twilio_service as svc
        importlib.reload(svc)

        fake_http_client = MagicMock()
        fake_twilio_client = MagicMock()

        env = {
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "tokentest",
        }

        with patch.dict(os.environ, env, clear=True):
            with mock_patch("services.twilio_service.TwilioHttpClient", return_value=fake_http_client) as mock_http_cls:
                with mock_patch("services.twilio_service.Client", return_value=fake_twilio_client) as mock_client_cls:
                    result = svc._get_client()

        mock_http_cls.assert_called_once_with(timeout=10)
        call_kwargs = mock_client_cls.call_args[1]
        self.assertIs(call_kwargs["http_client"], fake_http_client)
        self.assertIs(result, fake_twilio_client)


# ---------------------------------------------------------------------------
# J. Live-mode config error cases
# ---------------------------------------------------------------------------
class TestLiveModeConfigErrors(unittest.TestCase):

    def test_missing_account_sid_raises_config_error(self):
        import services.twilio_service as svc
        importlib.reload(svc)

        from models import SMSPayload
        from services.twilio_service import SMSSendError
        payload = SMSPayload(to_phone="+14155550192", message="test", lead_score=50)

        env = {
            "TWILIO_DRY_RUN": "false",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_PHONE_NUMBER": "+14155550199",
            # TWILIO_ACCOUNT_SID intentionally absent
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SMSSendError) as ctx:
                svc.send_sms(payload)
        self.assertEqual(ctx.exception.category, "config")

    def test_invalid_phone_number_in_env_raises_config_error(self):
        import services.twilio_service as svc
        importlib.reload(svc)

        from models import SMSPayload
        from services.twilio_service import SMSSendError
        payload = SMSPayload(to_phone="+14155550192", message="test", lead_score=50)

        env = {
            "TWILIO_DRY_RUN": "false",
            "TWILIO_ACCOUNT_SID": "ACtest",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_PHONE_NUMBER": "not-a-number",
        }
        with patch.dict(os.environ, env, clear=True):
            with mock_patch("services.twilio_service.Client"):
                with mock_patch("services.twilio_service.TwilioHttpClient"):
                    with self.assertRaises(SMSSendError) as ctx:
                        svc.send_sms(payload)
        self.assertEqual(ctx.exception.category, "config")


if __name__ == "__main__":
    unittest.main()
