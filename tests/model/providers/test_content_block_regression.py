"""Regression test: OpenRouter content-block errors should be scorable, not fatal.

This test imports only types that exist on base (OpenRouterAPI, OpenRouterError).
On base, content-block errors raise generic OpenRouterError (aborts sample).
On head, they raise OpenRouterContentBlockError (caught -> content_filter output).
"""

import pytest

from inspect_ai.model._providers.openrouter import OpenRouterAPI, OpenRouterError


def _make_api():
    return OpenRouterAPI(model_name="openrouter/google/gemini-flash", api_key="test")


def test_moderation_error_is_scorable_not_generic():
    """Test that a 403 moderation error is scorable.

    A 403 moderation error with metadata.reasons should signal a scorable
    content block, not a generic OpenRouterError that aborts the sample.
    """
    api = _make_api()
    err = {"code": 403, "message": "flagged", "metadata": {"reasons": ["harmful"]}}
    with pytest.raises(OpenRouterError) as exc_info:
        api.on_response({"error": err})
    # The exception should be a specific content-block error, not generic
    exc = exc_info.value
    assert type(exc).__name__ != "OpenRouterError", (
        f"Content-block moderation error raised generic OpenRouterError "
        f"(would abort sample). Expected a content-block-specific exception "
        f"that is caught and converted to content_filter output. Got: {type(exc).__name__}"
    )


def test_provider_block_error_is_scorable_not_generic():
    """Test that a 400 provider block error is scorable.

    A 400 provider block ('<Provider> blocked the request: REASON') should
    signal a scorable content block, not a generic OpenRouterError.
    """
    api = _make_api()
    err = {"code": 400, "message": "Gemini blocked the request: PROHIBITED_CONTENT"}
    with pytest.raises(OpenRouterError) as exc_info:
        api.on_response({"error": err})
    exc = exc_info.value
    assert type(exc).__name__ != "OpenRouterError", (
        f"Provider block error raised generic OpenRouterError "
        f"(would abort sample). Expected a content-block-specific exception. "
        f"Got: {type(exc).__name__}"
    )
