"""Provider construction from settings."""

from __future__ import annotations

import logging

from audit_agent.config import Settings, get_settings
from audit_agent.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Instantiate the configured LLM backend.

    Imports are deferred so the unselected provider's dependencies stay optional.
    """
    settings = settings or get_settings()

    if settings.llm_provider == "anthropic":
        from audit_agent.llm.anthropic_provider import AnthropicProvider

        logger.debug("Using Anthropic provider (%s)", settings.anthropic_model)
        return AnthropicProvider(
            api_key=settings.anthropic_api_key or "",
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
        )

    if settings.llm_provider == "ollama":
        from audit_agent.llm.ollama_provider import OllamaProvider

        logger.debug("Using Ollama provider (%s)", settings.ollama_model)
        return OllamaProvider(model=settings.ollama_model, temperature=settings.temperature)

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
