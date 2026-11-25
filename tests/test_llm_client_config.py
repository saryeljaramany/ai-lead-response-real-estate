"""
tests/test_llm_client_config.py

Offline unit tests for OpenRouter client timeout and retry configuration in
services/llm_service.py.

Pure ASCII, UTF-8 without BOM, LF only.
"""

import os
import unittest
from unittest.mock import patch

import services.llm_service as llm_service


class TestLlmClientConfig(unittest.TestCase):
    """Test suite for _get_client() configuration and timeout constants."""

    def test_request_timeout_constant_is_60(self) -> None:
        """_REQUEST_TIMEOUT_SECONDS is set to 60 seconds."""
        self.assertEqual(llm_service._REQUEST_TIMEOUT_SECONDS, 60)

    def test_get_client_configures_timeout_and_zero_retries(self) -> None:
        """_get_client() sets timeout=60, max_retries=0, base_url, api_key, and X-Title."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}, clear=True):
            with patch("services.llm_service.OpenAI") as mock_openai:
                client = llm_service._get_client()

        mock_openai.assert_called_once()
        kwargs = mock_openai.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(kwargs["api_key"], "test")
        self.assertIn("X-Title", kwargs["default_headers"])
        self.assertIs(client, mock_openai.return_value)

    def test_get_client_raises_runtime_error_when_api_key_missing(self) -> None:
        """Empty environment raises RuntimeError naming OPENROUTER_API_KEY without calling OpenAI."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("services.llm_service.OpenAI") as mock_openai:
                with self.assertRaises(RuntimeError) as ctx:
                    llm_service._get_client()

        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        mock_openai.assert_not_called()
