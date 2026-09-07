"""Local Ollama provider, for running the pipeline with no API cost or egress."""

from __future__ import annotations

import logging

from audit_agent.llm.base import ToolDefinition, ToolRequest, ToolUseNotSupportedError

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Local model backend.

    Useful when client evidence cannot leave the machine, at the cost of the
    tool-use routing the hosted provider supports.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
    ) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The ollama package is required for the local provider. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        self._ollama = ollama
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self._ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": self._temperature},
        )
        return str(response["message"]["content"])

    def supports_tools(self) -> bool:
        return False

    def decide_tool_use(self, user: str, tools: list[ToolDefinition]) -> ToolRequest:
        raise ToolUseNotSupportedError(
            "The Ollama provider does not implement tool routing. "
            "Use the Anthropic provider for the tool-use agent."
        )
