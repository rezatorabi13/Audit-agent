"""Local Ollama provider: the default backend, with no API cost or data egress."""

from __future__ import annotations

import json
import logging
from typing import Any

from audit_agent.llm.base import ToolDefinition, ToolRequest

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Local model backend (Llama 3.2 by default).

    Suited to engagements where client evidence cannot leave the machine.
    Supports tool use through Ollama's native tool-calling API, which Llama 3.2
    was trained for.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import ollama
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "The ollama package is required for the local provider. "
                    "Install it with: pip install -r requirements.txt"
                ) from exc
            client = ollama

        # ``client`` is the ollama module or anything exposing ``chat``; tests
        # inject a fake so no Ollama server is needed.
        self._ollama = client
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
        return str(_field(_field(response, "message"), "content") or "")

    def supports_tools(self) -> bool:
        return True

    def decide_tool_use(self, user: str, tools: list[ToolDefinition]) -> ToolRequest:
        """Ask the model whether the question warrants calling a tool.

        Small local models call tools more eagerly than hosted ones, so the
        system prompt states explicitly when a tool is *not* needed.
        """
        response = self._ollama.chat(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You can call tools. Call a tool only when the question needs "
                        "information you do not already have, such as client-specific "
                        "documents. For general knowledge or arithmetic, answer directly "
                        "without calling any tool."
                    ),
                },
                {"role": "user", "content": user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ],
            options={"temperature": self._temperature},
        )

        message = _field(response, "message")
        text = str(_field(message, "content") or "")
        tool_calls = _field(message, "tool_calls") or []
        known = {tool.name for tool in tools}

        for call in tool_calls:
            function = _field(call, "function")
            name = _field(function, "name")
            if name not in known:
                logger.warning("Model requested unknown tool %r; ignoring", name)
                continue
            return ToolRequest(
                tool_name=str(name),
                tool_input=_parse_arguments(_field(function, "arguments")),
                text=text,
            )

        return ToolRequest(tool_name=None, tool_input={}, text=text)


def _field(obj: Any, key: str) -> Any:
    """Read a field from an ollama response (a pydantic object or a plain dict)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Tool arguments arrive as a mapping, or as a JSON string on some versions."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return dict(arguments) if isinstance(arguments, dict) else {}
