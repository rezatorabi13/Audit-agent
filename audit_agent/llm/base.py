"""Provider-agnostic LLM interface.

Workflow nodes depend on this protocol rather than a concrete SDK, so the
hosted and local backends are interchangeable and tests can inject a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool the model may choose to call."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A model's decision to call a tool, or its direct answer if it declined."""

    tool_name: str | None
    tool_input: dict[str, Any]
    text: str

    @property
    def wants_tool(self) -> bool:
        return self.tool_name is not None


@runtime_checkable
class LLMProvider(Protocol):
    """Minimum surface the workflow needs from a language model."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return a single completion for a system/user prompt pair."""
        ...

    def supports_tools(self) -> bool:
        """Whether :meth:`decide_tool_use` is implemented by this provider."""
        ...

    def decide_tool_use(self, user: str, tools: list[ToolDefinition]) -> ToolRequest:
        """Let the model choose whether to call a tool before answering."""
        ...


class ToolUseNotSupportedError(NotImplementedError):
    """Raised when a provider without tool support is asked to route a tool call."""
